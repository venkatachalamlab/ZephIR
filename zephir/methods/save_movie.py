from ..utils.utils import *


def save_movie(
    container,
    results,):
    """Visualize results in an annotated movie.

    Visualize results as coloured annotations in a movie. Frames are generated
    using the same data loading parameters used for tracking.

    :param container: variable container, needs to contain: dataset, channel,
    exclude_self, exclusive_prov, gamma, include_all, img_shape, shape_t, worldline_id
    :param results: tracking results
    """

    # pull variables from container
    dataset = container.get('dataset')
    channel = container.get('channel')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    gamma = container.get('gamma')
    include_all = container.get('include_all')
    img_shape = container.get('img_shape')
    shape_t = container.get('shape_t')
    t_annot = container.get('t_annot')
    t_list = container.get('t_list')
    worldline_id = container.get('worldline_id')

    annotation = get_annotation_df(dataset)

    # visualizing result and writing to video
    print('\nSaving prediction to video...')
    overview_rec = cv2.VideoWriter(
            str(dataset / 'annotated.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), 10,
            (img_shape[2] + 2 * img_shape[0],
             img_shape[1] + 2 * img_shape[0]),
            True
    )

    if include_all:
        t_list = np.arange(shape_t)
    else:
        t_list = np.unique(list(t_annot) + list(t_list))

    for t in tqdm(t_list, desc='Saving to video', unit='frames'):
        data = get_data(dataset, t, g=gamma, c=channel)
        xyz_result = results[t, ...]
        if include_all and t in np.unique(annotation['t_idx']):
            u, annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)
            w_idx = np.where(np.isin(worldline_id, u))[0]
            if len(w_idx) > 0:
                u_idx = np.where(np.isin(u, worldline_id))[0]
                xyz_result[w_idx, ...] = annot[u_idx, ...]
            if len(u) > len(w_idx):
                u_idx = np.where(np.isin(u, worldline_id, invert=True))[0]
                xyz_result = np.append(xyz_result, annot[u_idx, ...], axis=0)
        overview = save_as_bgr(data, annotation=xyz_result)
        cv2.putText(overview, f'Frame #{t}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
        overview_rec.write(overview[:, :, ::-1])
    overview_rec.release()

    return
