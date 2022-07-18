import sklearn.neighbors as sk

from ..utils.utils import *


def build_springs(
    container,
    load_nn,
    nn_max,
    verbose=False,):
    """Build nearest-neighbor spring network for spatial regularization.

    Build KDTree of nn_max nearest neighbors around each annotation. These connections
    are used to calculate spatial regularization loss, L_N. Covariances of
    connected pairs are compiled if multiple reference frames are available.

    :param container: variable container, needs to contain: dataset, annot,
    grid_shape, img_shape, shape_n, t_annot, verbose, z_compensator
    :param load_nn: load or save existing nn_idx.txt for spring connection indices
    :param nn_max: maximum number of neighbors to connect to for each keypoint
    :return: container (updated entries for: covar, neighbors)
    """

    # pull variables from container
    dataset = container.get('dataset')
    annotations = container.get('annot')
    grid_shape = container.get('grid_shape')
    img_shape = container.get('img_shape')
    shape_n = container.get('shape_n')
    t_annot = container.get('t_annot')
    z_compensator = container.get('z_compensator')

    print('\nBuilding neighbor tree...')
    d_scaler = np.array([1, 1, z_compensator])
    weighted_shape = np.array(img_shape)[::-1] * np.array([1, 1, z_compensator])

    covar = None
    if len(t_annot) > 1:
        # getting variance of intra-keypoint distances from annotations
        covar = np.zeros((shape_n, shape_n))
        for dim in range(3):
            covar += np.cov(
                (np.array(annotations)[:, :, dim]+1)/2 * weighted_shape[dim],
                rowvar=False) / 3

        if verbose:
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle(f'mean={np.mean(covar):.3f}  '
                         f'max={np.max(covar):.3f}  '
                         f'min={np.min(covar):.3f}', fontsize=14)
            c = covar + np.min(covar)
            plt.imshow(c / np.max(c))
            plt.show()
            fig.savefig(str(dataset / 'pw_covariance.png'))

    # neighbors are filtered by position covariance
    nn_max = min(nn_max, shape_n-1)
    if load_nn and (dataset / 'nn_idx.txt').is_file():
        neighbors = []
        with open(str(dataset / 'nn_idx.txt'), 'r') as k:
            nn_idx = [line.rstrip().split(' ') for line in k.readlines()]
        nn_max = np.max([len(k) for k in nn_idx])
        ind = np.tile(np.arange(shape_n)[:, None], (1, nn_max))
        for nn in nn_idx:
            ind[int(nn[0]), :len(nn[1:])] = np.array(nn[1:]).astype(int)
        for annot in annotations:
            d_ref = (np.concatenate([annot[ind[:, k], np.newaxis, :]
                                     for k in range(nn_max)], axis=1)
                     - np.tile(annot[:, np.newaxis, :], (1, nn_max, 1)))
            neighbors.append(np.append(ind[:, :, np.newaxis], d_ref, axis=-1))

    elif nn_max > 0:
        neighbors = []
        for annot in annotations:
            weighted_xyz = (annot + 1)/2 * weighted_shape
            tree = sk.KDTree(weighted_xyz)
            dist, ind = tree.query(weighted_xyz, k=nn_max+1, return_distance=True)
            ind = np.where(
                dist[:, 1:] < 1.6/np.prod(d_scaler) * np.linalg.norm(grid_shape*d_scaler),
                ind[:, 1:],
                np.tile(np.arange(shape_n)[:, None], (1, nn_max))
            )
            d_ref = (np.concatenate([annot[ind[:, k], np.newaxis, :]
                                     for k in range(nn_max)], axis=1)
                     - np.tile(annot[:, np.newaxis, :], (1, nn_max, 1)))
            neighbors.append(np.append(ind[:, :, np.newaxis], d_ref, axis=-1))
            if load_nn:
                np.savetxt(
                    str(dataset / 'nn_idx.txt'),
                    np.append(np.arange(shape_n)[:, None], ind, axis=-1),
                    fmt='%d'
                )

    else:
        neighbors = [np.tile(np.arange(0, shape_n)[:, None], (1, nn_max))
                     for _ in t_annot]

    if verbose:
        data = get_data(dataset, t_annot[0])
        overview = save_as_bgr(data, annotation=annotations[0])
        for n in range(shape_n):
            xn, yn, _ = get_pixel(annotations[0][n, ...], img_shape)
            ind = neighbors[0][n, :, 0].astype(int)
            for i in ind:
                xi, yi, _ = get_pixel(annotations[0][i, ...], img_shape)
                overview = cv2.line(overview, (xn, yn), (xi, yi),
                                    color=pick_color(n))
        fig = plt.figure(dpi=600)
        plt.imshow(overview)
        plt.show()
        fig.savefig(str(dataset / f'skeleton #{t_annot[0]}.png'))

    # push variables to container
    container.update({
        'covar': covar,
        'neighbors': neighbors,
    })

    return container
