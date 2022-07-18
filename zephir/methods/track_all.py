from collections import OrderedDict
import copy
import torch.nn as nn
import torch.optim as optim
from itertools import product
from scipy.interpolate import griddata

from ..models.losses import *
from ..utils.utils import *


def track_all(
    container,
    results,
    zephir,
    zephod,
    clip_grad,
    lambda_t,
    lambda_d,
    lambda_n,
    lambda_n_mode,
    lr_ceiling,
    lr_coef,
    lr_floor,
    motion_predict,
    n_epoch,
    n_epoch_d,
    kernel_size=(3, 9, 9),
    lambda_n_decay=1.0,
    lr_step_size=10,
    lr_gamma=0.5,
    nb_delta=(2, 1),
    nb_epoch=5,
    restrict_update=False,
    sigmas=(1, 4, 4),
    _t_list=None,):

    # pull variables from container
    dataset = container.get('dataset')
    allow_rotation = container.get('allow_rotation')
    annot = container.get('annot')
    channel = container.get('channel')
    covar = container.get('covar')
    dev = container.get('dev')
    gamma = container.get('gamma')
    n_frame = container.get('n_frame')
    neighbors = container.get('neighbors')
    partial_annot = container.get('partial_annot')
    p_list = container.get('p_list')
    r_list = container.get('r_list')
    s_list = container.get('s_list')
    shape_t = container.get('shape_t')
    t_annot = container.get('t_annot')
    t_list = container.get('t_list')
    z_compensator = container.get('z_compensator')

    if _t_list is None:
        _t_list = t_list.copy()
        update_checkpoint(dataset, {
            'results': results,
            '_t_list': _t_list,
        }, verbose=False)
    else:
        t_list = _t_list.copy()

    zephir.to(dev)
    tpbar = tqdm(t_list, desc='Analyzing movie', unit='frames')
    for t in tpbar:

        # get frame branch information
        parent = p_list[t]
        root = r_list[t]
        if parent < 0:
            # root frame, does not need registration
            results[t] = annot[root]
            continue

        # calculate initial learning rate based on parent-child distance
        distance = s_list[t]
        lr_init = min(max(lr_coef * distance, lr_floor), lr_ceiling)
        tqdm.write(f'\nFrame #{t}\t\tParent #{parent}'
                   f'\t\tReference #{t_annot[root]}'
                   f'\t\tDistance to parent: d={distance:.4f}')

        if n_frame > 1:
            t_patch = np.arange(max(t-n_frame//2, 0), min(t+n_frame//2+1, shape_t))
            t_idx = np.where(t_patch == t)[0][0]
        else:
            t_patch = np.array([t])
            t_idx = 0

        with torch.no_grad():

            # compiling Zephod results for feature detection loss, L_D
            input_tensor_d = []
            if lambda_d > 0 and zephod is not None:
                for _t in t_patch:
                    data = get_data(dataset, _t, g=gamma, c=channel)
                    _pred = zephod(data)
                    _pred = (torch.max(_pred) - _pred) / torch.max(_pred)
                    input_tensor_d.append(torch.where(_pred < 0.5, 0., 0.5))
                input_tensor_d = torch.stack(input_tensor_d, dim=0)
                input_tensor_d = blur3d(input_tensor_d, (1, 5, 5), (1, 2, 2), dev=dev)
                input_tensor_d.requires_grad = False

            # compiling target descriptors from reference frame for image registration, L_R
            data = get_data(dataset, t_annot[root], g=gamma, c=channel)
            vol = to_tensor(data, n_dim=5, dev=dev)
            zephir.theta.zero_()
            zephir.rho.zero_()
            zephir.rho[:len(t_patch)].add_(
                to_tensor(annot[root], dev=dev).expand(len(t_patch), -1, -1)
            )
            target_tensor = zephir(vol.expand(len(t_patch), -1, -1, -1, -1, -1))

            # loading input volumes
            input_tensor = []
            for _t in t_patch:
                data = get_data(dataset, _t, g=gamma, c=channel)
                vol = to_tensor(data, n_dim=5, grad=False, dev=dev)
                input_tensor.append(vol)
            input_tensor = torch.stack(input_tensor, dim=0)

            # initializing model parameters at parent coordinates
            xyz_parent = to_tensor(results[parent, ...], dev=dev)
            zephir.theta.zero_()
            zephir.rho.zero_()
            zephir.rho[:len(t_patch)].add_(
                xyz_parent.expand(len(t_patch), -1, -1)
            )

            # compiling spring connections and stiffnesses for spatial regularization, L_N
            ind = neighbors[root][:, :, 0]
            d_ref = to_tensor(neighbors[root][:, :, 1:], dev=dev)
            if covar is not None:
                nn_covar = np.empty_like(ind)
                for k in range(ind.shape[1]):
                    nn_covar[:, k] = covar[range(ind.shape[0]), ind[:, k].astype(int)]/20
                k_ij = to_tensor(
                    nn_covar /
                    np.clip(np.max(nn_covar, axis=-1)[:, None], 1, None),
                    grad=False, dev=dev
                )
                k_ij = lambda_n * torch.relu(k_ij)
            else:
                k_ij = lambda_n * torch.ones_like(
                    to_tensor(ind), requires_grad=False, device=dev
                )

            # loading partial annotations if available
            p_idx, pins, params = [], [], []
            subset = _rho = None
            for i, _t in enumerate(t_patch):
                if _t in partial_annot.keys():
                    tqdm.write(f'*** Partial annotations available for frame {_t}')
                    _pins, _params = partial_annot[_t]

                    if motion_predict:
                        # using partial annotations to interpolate a flow field
                        # and calculate new initial parameters for tracking
                        corners = np.stack(list(product((-1, 1), repeat=3)), axis=0)
                        p = np.append(results[parent, _pins, :], corners, axis=0)
                        v = np.append(_params - results[parent, _pins, :], np.zeros((8, 3)), axis=0)
                        gd = griddata(p, v, results[parent, ...], method='linear', fill_value=0)
                        zephir.rho[i].add_(to_tensor(gd, dev=dev))

                    p_idx.append(i)
                    pins.append(to_tensor(_pins, dtype=torch.int64, dev=dev))
                    params.append(to_tensor(_params, dev=dev))

            if restrict_update and t in partial_annot.keys():
                # restricting tracking to only keypoints neighboring partial annotations
                _pins, _params = partial_annot[t]
                subset = np.unique(np.append(_pins, ind[_pins, :])).astype(int)
                tqdm.write(f'*** RESTRICT_UPDATE ACTIVE. '
                           f'Deepcopying model with restricted parameters '
                           f'for {len(subset)} keypoints...')
                zephir = copy.deepcopy(zephir)
                zephir.shape_n = len(subset)
                zephir.grid = nn.Parameter(zephir.grid[:, subset], requires_grad=False)
                zephir.rho = nn.Parameter(zephir.rho[:, subset], requires_grad=True)
                zephir.theta = nn.Parameter(zephir.theta[:, subset], requires_grad=allow_rotation)
                target_tensor = target_tensor[:, subset]
                _rho = to_tensor(results[t_patch], dev=dev, grad=False)
                _rho[:, subset] = zephir.rho.clone()
                _rho.requires_grad = False

            # resetting optimizer state
            optimizer = optim.SGD(zephir.parameters(), lr=lr_init)
            scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step_size, lr_gamma)
            _kernel_size = kernel_size
            _sigmas = sigmas

        # training loop here
        zephir.train()
        pbar = tqdm(range(n_epoch + n_epoch_d), desc='Tracking', unit='epochs')
        for epoch in pbar:

            if (epoch + 1) % nb_epoch == 0:
                with torch.no_grad():
                    # updating spring stiffness
                    if lambda_n_decay < 1.0:
                        k_ij *= lambda_n_decay
                    # updating kernels for dynamic Gaussian blurring
                    _kernel_size = (3, _kernel_size[1] - nb_delta[0], _kernel_size[2] - nb_delta[0])
                    _sigmas = (1, _sigmas[1] - nb_delta[1], _sigmas[1] - nb_delta[1])
                    if _kernel_size[1] <= 1 or _sigmas[1] <= 1:
                        # resets kernels to be inactive when it reaches a threshold
                        _kernel_size = (1, 1, 1)
                        _sigmas = (1, 1, 1)

            optimizer.zero_grad()

            # using automatic matrix precision casting to dynamically switch
            # between float16 and float32 to minimize memory consumption
            with torch.autocast(device_type=dev):

                # sampling child descriptors at model parameters from child frames
                pred = zephir(input_tensor)

                # dynamic Gaussian blurring
                blur_target = blur3d(target_tensor, _kernel_size, _sigmas, dev=dev)
                blur_pred = blur3d(pred, _kernel_size, _sigmas, dev=dev)

                if epoch > n_epoch and lambda_d > 0 and input_tensor_d is not None:
                    # feature detection loss, L_D
                    loss = torch.mean(reg_d(
                        lambda_d, input_tensor_d,
                        zephir.rho, zephir.n_chunks
                    ))
                else:
                    # image registration loss, L_R
                    loss = torch.mean(corr_loss(blur_pred, blur_target))

                    reg = 0
                    if torch.max(k_ij) > 0:
                        # spatial regularization, L_N
                        reg += torch.mean(
                            reg_n(k_ij, zephir.rho, _rho, ind, d_ref,
                                  subset, lambda_n_mode)
                        )
                    if lambda_t > 0:
                        # temporal smoothing, L_T
                        reg += torch.mean(reg_t(lambda_t, blur_pred, (1, 5, 5), None))
                    loss += reg

            # backpropagation call
            loss.backward()
            if clip_grad > 0.:
                nn.utils.clip_grad_value_(zephir.parameters(), clip_grad)
            if z_compensator > 0.:
                # due to anisotropy in depth-axis commonly found in 3D datasets,
                # gradient descent may not properly update parameters in that axis
                # so we multiply calculated gradients in that axis by (1+z_compensator)
                # to compensate
                with torch.no_grad():
                    zephir.rho[:, :, -1] += (
                        - z_compensator
                        * optimizer.param_groups[0]['lr']
                        * zephir.rho.grad[:, :, -1]
                    )

            # gradient descent step, update model parameters
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(OrderedDict(
                    Loss=f'{loss.item():.4f}',
                    Loss_R=f'{(loss - reg).item():.4f}',
                    LR=f'{current_lr:.4f}'
                ))

                # resetting model parameters for partial annotations
                if len(p_idx) > 0:
                    for _i, i in enumerate(p_idx):
                        _pins, _params = pins[_i], params[_i]
                        if subset is not None:
                            _pins = [np.where(subset == w)[0][-1]
                                     for w in to_numpy(_pins) if w in subset]
                            _pins = to_tensor(_pins, dtype=torch.int64, dev=dev)
                        zephir.rho[i].index_fill_(0, _pins, 0)
                        zephir.rho[i].index_add_(0, _pins, _params)

                if subset is not None:
                    _rho[:, subset] = zephir.rho.clone()
                    _rho.requires_grad = False
                
                # resetting model parameters for rotation
                if not allow_rotation:
                    zephir.theta.zero_()

        # updating array with tracking results for the child frame
        with torch.no_grad():
            if subset is not None:
                _rho[t_idx, subset, :] = zephir.rho[t_idx]
                results[t] = to_numpy(_rho[t_idx])
            else:
                results[t] = to_numpy(zephir.rho)[t_idx]

            _t_list = np.setdiff1d(_t_list, np.array([t]), assume_unique=True)

            # push results to checkpoint
            update_checkpoint(dataset, {
                'results': results,
                '_t_list': _t_list,
            }, verbose=False)

    return container, results
