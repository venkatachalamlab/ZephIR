from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from .build_pdists import get_all_pdists
from ..models.losses import *
from ..utils.utils import *


def get_optimization_trajectory(
    frame_to_optimize,
    parent,
    reference,
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
    n_epoch,
    n_epoch_d,
    kernel_size=(3, 9, 9),
    lambda_n_decay=1.0,
    lr_step_size=10,
    lr_gamma=0.5,
    nb_delta=(2, 1),
    nb_epoch=5,
    sigmas=(1, 4, 4),):

    # pull variables from container
    dataset = container.get('dataset')
    allow_rotation = container.get('allow_rotation')
    annot = container.get('annot')
    channel = container.get('channel')
    covar = container.get('covar')
    dev = container.get('dev')
    gamma = container.get('gamma')
    img_shape = container.get('img_shape')
    neighbors = container.get('neighbors')
    shape_t = container.get('shape_t')
    t_annot = container.get('t_annot')
    z_compensator = container.get('z_compensator')

    trajectory = np.zeros((n_epoch+n_epoch_d, *results.shape[1:]))

    zephir.to(dev)

    t = frame_to_optimize
    root = np.where(t_annot == reference)[0][0]

    # calculate initial learning rate based on parent-child distance
    d_full = get_all_pdists(dataset, shape_t, channel, pbar=True)
    distance = d_full[t, parent]
    lr_init = min(max(lr_coef * distance, lr_floor), lr_ceiling)
    tqdm.write(f'\nFrame #{t}\t\tParent #{parent}'
               f'\t\tReference #{t_annot[root]}'
               f'\t\tDistance to parent: d={distance:.4f}')

    t_patch = [t]

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
                        reg_n(k_ij, zephir.rho, None, ind, d_ref,
                              None, lambda_n_mode)
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

            # resetting model parameters for rotation
            if not allow_rotation:
                zephir.theta.zero_()

            for n in range(zephir.rho.shape[1]):
                x, y, z = get_pixel(zephir.rho.clone()[0, n, :], img_shape)
                trajectory[epoch, n, :] = np.array([x, y, z])

    return trajectory
