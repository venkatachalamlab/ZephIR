from tqdm.notebook import tqdm

from ..models.zephir import ZephIR
from ..models.losses import *
from ..utils.utils import *


def plot_loss_maps(
    keypoint_to_visualize,
    frame_to_visualize,
    reference,
    map_resolution,
    map_size,
    trajectory,
    container,
    dimmer_ratio,
    grid_shape,
    fovea_sigma,
    n_chunks,
    zephod,
    lambda_d,
    lambda_n,
    lambda_n_mode,):

    # pull variables from container
    dataset = container.get('dataset')
    annot = container.get('annot')
    channel = container.get('channel')
    covar = container.get('covar')
    dev = container.get('dev')
    gamma = container.get('gamma')
    img_shape = np.array(container.get('img_shape'))
    neighbors = container.get('neighbors')
    shape_n = container.get('shape_n')
    t_annot = container.get('t_annot')

    z_stacks = min(img_shape[0], grid_shape[0])
    grid_shape = (z_stacks - (z_stacks+1) % 2, grid_shape[1], grid_shape[2])
    grid_spacing = tuple(np.array(grid_shape) / img_shape)
    model_kwargs = {
        'allow_rotation': False,
        'dimmer_ratio': dimmer_ratio,
        'fovea_sigma': fovea_sigma,
        'grid_shape': grid_shape,
        'grid_spacing': grid_spacing,
        'n_chunks': n_chunks,
        'n_frame': 2,
        'shape_n': shape_n,
        'ftr_ratio': 0.6,
        'ret_stride': 2,
    }
    zephir = ZephIR(**model_kwargs).to(dev)

    t = frame_to_visualize
    root = np.where(t_annot == reference)[0][0]
    target = np.where(t_annot == frame_to_visualize)[0][0]

    t_patch = [t, reference]

    xv, yv = np.meshgrid(
        np.linspace(-map_size//2, map_size//2, map_resolution),
        np.linspace(-map_size//2, map_size//2, map_resolution)
    )

    losses_at_trajectory = np.zeros((map_resolution, map_resolution, 4))
    losses_at_annotation = losses_at_trajectory.copy()

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

        # compiling spring connections and stiffnesses for spatial regularization, L_N
        ind = neighbors[root][:, :, 0]
        d_ref = to_tensor(neighbors[root][:, :, 1:], dev=dev)
        if covar is not None:
            nn_covar = np.empty_like(ind)
            for k in range(ind.shape[1]):
                nn_covar[:, k] = covar[range(ind.shape[0]), ind[:, k].astype(int)] / 20
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

        for i in tqdm(range(map_resolution), desc='Calculating loss map', unit='row'):
            for j in range(map_resolution):
                dx, dy = xv[i, j], yv[i, j]

                rho_at_prediction = trajectory[-1, ...] / (img_shape[::-1])[None, :] * 2 - 1
                rho_at_prediction = to_tensor(rho_at_prediction, n_dim=3, dev=dev)
                rho_at_prediction[0, keypoint_to_visualize, 0] += dx / img_shape[2] * 2
                rho_at_prediction[0, keypoint_to_visualize, 1] += dy / img_shape[1] * 2

                rho_at_annotation = to_tensor(annot[target], n_dim=3, dev=dev)
                rho_at_annotation[0, keypoint_to_visualize, 0] += dx / img_shape[2] * 2
                rho_at_annotation[0, keypoint_to_visualize, 1] += dy / img_shape[1] * 2

                zephir.rho.zero_()
                zephir.rho.add_(torch.cat([rho_at_prediction, rho_at_annotation], dim=0))
                zephir.theta.zero_()

                pred = zephir(input_tensor)

                loss_r = corr_loss(pred, target_tensor)
                loss_n = reg_n(k_ij, zephir.rho, None, ind, d_ref, None, lambda_n_mode)
                loss_d = reg_d(lambda_d, input_tensor_d, zephir.rho, zephir.n_chunks)

                losses_at_trajectory[i, j, :] = np.array([
                    loss_r[0].item(), loss_n[0].item(), loss_d[0].item(),
                    loss_r[0].item() + loss_n[0].item() + loss_d[0].item()
                ])
                losses_at_annotation[i, j, :] = np.array([
                    loss_r[1].item(), loss_n[1].item(), loss_d[1].item(),
                    loss_r[1].item() + loss_n[1].item() + loss_d[1].item()
                ])

    # plot around predicted position
    xyz_at_prediction = trajectory[-1, keypoint_to_visualize, :].astype(int)
    rho_at_annotation = to_tensor(annot[target], dev=dev)
    xyz_at_annotation = np.array(
        get_pixel(rho_at_annotation[keypoint_to_visualize, :],
                  img_shape)
    )

    print(f'{keypoint_to_visualize}'
          f'\tinit: {list(trajectory[0, keypoint_to_visualize, :])}'
          f'\tfinal: {list(trajectory[-1, keypoint_to_visualize, :])}'
          f'\ttarget: {list(xyz_at_annotation)}')

    xy_target_at_prediction = map_resolution/map_size * (
        xyz_at_annotation[:2] - xyz_at_prediction[:2]
    ) + map_resolution/2
    xy_trajectory_at_prediction = map_resolution/map_size * (
        trajectory[:, keypoint_to_visualize, :2] - xyz_at_prediction[:2]
    ) + map_resolution/2

    volume_crop_at_trajectory = np.max(
        to_numpy(
            input_tensor[
             0, 0, 0,
             max(xyz_at_prediction[2] - 3, 0):xyz_at_prediction[2] + 3,
             max(xyz_at_prediction[1] - map_resolution//2, 0):xyz_at_prediction[1] + map_resolution//2,
             max(xyz_at_prediction[0] - map_resolution//2, 0):xyz_at_prediction[0] + map_resolution//2
            ]
        ), axis=0
    )

    fig, axes = plt.subplots(2, 5, constrained_layout=True, figsize=(25, 10))

    for i in range(5):
        axes[0, i].scatter(xy_trajectory_at_prediction[0, 0], xy_trajectory_at_prediction[0, 1], s=100.0, c='r')
        axes[0, i].scatter(xy_trajectory_at_prediction[-1, 0], xy_trajectory_at_prediction[-1, 1], s=100.0, c='g')
        axes[0, i].plot(xy_trajectory_at_prediction[:, 0], xy_trajectory_at_prediction[:, 1], linewidth=10.0, c='darkorange', alpha=0.3)
        axes[0, i].scatter(xy_target_at_prediction[0], xy_target_at_prediction[1], s=200.0, c='violet')
        if i == 0:
            axes[0, i].imshow(volume_crop_at_trajectory)
        else:
            axes[0, i].imshow(losses_at_trajectory[..., i-1])

    # plot around correct position
    xy_target_at_annotation = map_resolution/map_size * (
        xyz_at_annotation[:2] - xyz_at_annotation[:2]
    ) + map_resolution/2
    xy_trajectory_at_annotation = map_resolution/map_size * (
        trajectory[:, keypoint_to_visualize, :2] - xyz_at_annotation[:2]
    ) + map_resolution/2

    volume_crop_at_annotation = np.max(
        to_numpy(
            input_tensor[
             0, 0, 0,
             max(xyz_at_annotation[2] - 3, 0):xyz_at_annotation[2] + 3,
             max(xyz_at_annotation[1] - map_resolution//2, 0):xyz_at_annotation[1] + map_resolution//2,
             max(xyz_at_annotation[0] - map_resolution//2, 0):xyz_at_annotation[0] + map_resolution//2
            ]
        ), axis=0
    )

    for i in range(5):
        axes[1, i].scatter(xy_trajectory_at_annotation[0, 0], xy_trajectory_at_annotation[0, 1], s=100.0, c='r')
        axes[1, i].scatter(xy_trajectory_at_annotation[-1, 0], xy_trajectory_at_annotation[-1, 1], s=100.0, c='g')
        axes[1, i].plot(xy_trajectory_at_annotation[:, 0], xy_trajectory_at_annotation[:, 1], linewidth=10.0, c='darkorange', alpha=0.3)
        axes[1, i].scatter(xy_target_at_annotation[0], xy_target_at_annotation[1], s=200.0, c='violet')
        if i == 0:
            axes[1, i].imshow(volume_crop_at_annotation)
        else:
            axes[1, i].imshow(losses_at_annotation[..., i-1])

    plt.show()

    return losses_at_trajectory, losses_at_annotation
