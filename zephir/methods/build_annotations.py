from ..utils.utils import *


def build_annotations(
    container,
    annotation,
    t_ref,
    wlid_ref,
    n_ref,):
    """Load and handle annotations from annotations.h5 file.

    Annotations are loaded and sorted according to user arguments, and used to
    populate an empty results array. This array will be filled during tracking.

    :param container: variable container; needs to contain: dataset,
    exclude_self, exclusive_prov, shape_t
    :param annotation: override annotations to use instead of loading from file
    :param t_ref: override frames to use as annotations
    :param wlid_ref: override worldline id's to analyze
    :param n_ref: override maximum number of keypoints to analyze
    :return: container (updated entries for: annot, shape_n, partial_annot,
    provenance, t_annot, worldline_id), results (pre-filled with loaded annotations)
    """

    # pull variables from container
    dataset = container.get('dataset')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    shape_t = container.get('shape_t')

    # checking annotated frames
    if annotation is None:
        annotation = get_annotation_df(dataset)
    t_annot = np.unique(annotation['t_idx']).astype(int)
    if t_ref is not None:
        if type(t_ref) is int:
            t_ref = [t_ref]
        t_annot = np.array([t for t in t_ref if t in t_annot]).astype(int)
    t_annot = np.sort(t_annot)

    worldline_id = None
    if wlid_ref is not None:
        if type(wlid_ref) is int:
            worldline_id = np.arange(wlid_ref)
        elif type(wlid_ref) is tuple and len(wlid_ref) == 2:
            worldline_id = np.arange(min(wlid_ref), max(wlid_ref))
        elif type(wlid_ref) is tuple or type(wlid_ref) is list:
            worldline_id = np.sort(np.array(wlid_ref))
        shape_n = len(worldline_id)
    elif n_ref is not None:
        shape_n = n_ref
        for t in t_annot:
            u, _, _ = get_annotation(annotation, t, exclusive_prov, exclude_self)
            if len(u) == n_ref:
                worldline_id = u
                print(f'Using frame #{t} as initial reference with specified {n_ref} annotations...')
                break
    else:
        nn_list = [len(get_annotation(annotation, t, exclusive_prov, exclude_self)[0]) for t in t_annot]
        shape_n, t_max = np.max(nn_list), t_annot[np.argmax(nn_list)]
        worldline_id, _, _ = get_annotation(annotation, t_max, exclusive_prov, exclude_self)
        print(f'Using frame #{t_max} as initial reference with {shape_n} annotations found...')

    if shape_n == 0 or shape_n is None or worldline_id is None:
        print('\n******* ERROR: annotations could not be loaded properly! '
              'Check parameters: t_ref, wlid_ref, n_ref.\n\n')

    annot = []
    partial_annot = {}
    results = np.zeros((shape_t, shape_n, 3))
    provenance = np.array([[b'ZEIR'] * shape_n] * shape_t)

    for t in t_annot:
        # loading and sorting annotation by worldline_id
        u, _annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)

        # checking if worldlines are available in the annotation
        w_idx = np.array([np.where(worldline_id == w)[0][-1]
                          for w in u if w in worldline_id], dtype=int)
        u_idx = np.array([np.where(u == w)[0][-1]
                          for w in worldline_id if w in u], dtype=int)
        _annot = _annot[u_idx, ...]

        if _annot.shape[0] > shape_n or _annot.shape[0] == 0:
            t_annot = np.setdiff1d(t_annot, [t])
            continue
        elif _annot.shape[0] < shape_n:
            t_annot = np.setdiff1d(t_annot, [t])
            partial_annot[t] = (w_idx, _annot)
            results[t, w_idx] = _annot
            provenance[t, w_idx] = prov[u_idx]
            continue

        annot.append(_annot)
        results[t] = _annot
        provenance[t, w_idx] = prov[u_idx]

    print(f'\nAnnotations loaded for frames {list(t_annot)} '
          f'with shape: {np.array(annot).shape}')
    if len(partial_annot) > 0:
        print(f'*** Partial annotations found for {len(partial_annot)} frames: '
              f'{list(partial_annot.keys())}')

    # push variables to container
    container.update({
        'annot': annot,
        'shape_n': shape_n,
        'partial_annot': partial_annot,
        'provenance': provenance,
        't_annot': t_annot,
        'worldline_id': worldline_id,
    })

    # push results to checkpoint
    update_checkpoint(dataset, {
        'results': results,
    })

    return container, results
