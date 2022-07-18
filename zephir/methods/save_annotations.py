import shutil

from ..utils.utils import *


def save_annotations(
    container,
    results,
    save_mode,):
    """Save results to file.

    Handles tracking results and compiles to an annotations data frame to save
    to file according to save_mode: 'o' will overwrite existing annotations.h5,
    'w' will write to coordinates.h5. Existing annotations will overwrite
    ZephIR results if include_all is True.

    :param container: variable container, needs to contain: dataset, exclude_self,
    exclusive_prov, include_all, p_list, provenance, shape_t, shape_n, worldline_id
    :param results: tracking results
    :param save_mode: mode for writing to file
    """

    # pull variables from container
    dataset = container.get('dataset')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    include_all = container.get('include_all')
    p_list = container.get('p_list')
    provenance = container.get('provenance')
    shape_t = container.get('shape_t')
    shape_n = container.get('shape_n')
    worldline_id = container.get('worldline_id')

    annotation = get_annotation_df(dataset)
    
    # saving result to .h5
    print('\nCompiling and saving results to file...')
    p_list = np.array(p_list)
    p_list[np.where(p_list==-1)] = np.where(p_list==-1)
    xyz_pd = np.concatenate(
        (np.repeat(np.arange(shape_t), shape_n)[:, np.newaxis],
         results.reshape((-1, 3)) / 2.0 + 0.5,
         np.tile(worldline_id, shape_t)[:, np.newaxis],
         np.repeat(p_list, shape_n)[:, np.newaxis],
         provenance.reshape((-1, 1))),
        axis=-1
    )

    if include_all:
        for t in np.unique(annotation['t_idx']):
            u, annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)
            w_idx = np.where(
                np.logical_and(
                    xyz_pd[t * shape_n:(t+1) * shape_n, 0].astype(np.float32) == t,
                    np.isin(xyz_pd[t * shape_n:(t+1) * shape_n, 4].astype(np.float32), u))
            )[0] + t * shape_n
            if len(w_idx) > 0:
                u_idx = np.where(np.isin(u, xyz_pd[w_idx, 4].astype(np.float32)))[0]
                xyz_pd[w_idx, :] = np.concatenate(
                    (np.ones((len(u_idx), 1)) * t,
                     annot[u_idx, ...] / 2 + 0.5,
                     u[u_idx, np.newaxis],
                     np.ones((len(u_idx), 1)) * p_list[t],
                     prov[u_idx, np.newaxis]),
                    axis=-1
                )
            if exclusive_prov is not None or exclude_self is True:
                u, annot, prov = get_annotation(annotation, t, None, False)
                w_idx = np.where(
                    np.logical_and(
                        xyz_pd[t * shape_n:(t + 1) * shape_n, 0].astype(np.float32) == t,
                        np.isin(xyz_pd[t * shape_n:(t + 1) * shape_n, 4].astype(np.float32), u))
                )[0] + t * shape_n
            if len(u) > len(w_idx):
                u_idx = np.where(np.isin(u, xyz_pd[w_idx, 4].astype(np.float32), invert=True))[0]
                xyz_pd = np.append(
                    xyz_pd,
                    np.concatenate(
                        (np.ones((len(u_idx), 1)) * t,
                         annot[u_idx, ...] / 2 + 0.5,
                         u[u_idx, np.newaxis],
                         np.ones((len(u_idx), 1)) * p_list[t],
                         prov[u_idx, np.newaxis]),
                        axis=-1
                    ),
                    axis=0
                )

    columns = {
        't_idx': np.uint32,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'worldline_id': np.uint32,
        'parent_id': np.uint32,
        'provenance': np.dtype("S4"),
    }
    if save_mode == 'o':
        if not (dataset / 'backup').is_dir():
            Path.mkdir(dataset / 'backup')
        now = datetime.datetime.now()
        now_ = now.strftime("%m_%d_%Y_%H_%M")
        shutil.copy(dataset / 'annotations.h5',
                    dataset / 'backup' / f'annotations_{now_}.h5')
        f = h5py.File(dataset / 'annotations.h5', mode='w')
    else:
        f = h5py.File(dataset / 'coordinates.h5', mode=save_mode)

    data = np.array(list(range(1, xyz_pd.shape[0] + 1)), dtype=np.uint32)
    f.create_dataset('id', shape=(xyz_pd.shape[0], ), dtype=np.uint32, data=data)

    for i, c in enumerate(columns.keys()):
        if c == 'provenance':
            data = np.array(xyz_pd[:, i], dtype=columns[c])
        else:
            data = np.array(xyz_pd[:, i].astype(np.float32), dtype=columns[c])
        f.create_dataset(c, shape=(xyz_pd.shape[0], ), dtype=columns[c], data=data)

    f.close()

    return
