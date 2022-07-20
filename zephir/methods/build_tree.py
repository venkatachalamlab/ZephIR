from scipy.sparse.csgraph import shortest_path

from .build_pdists import get_all_pdists
from ..utils.utils import *


def build_tree(
    container,
    sort_mode,
    t_ignore,
    t_track,
    verbose=False):
    """Build frame tree.

    Build frame tree with branches according to given sort_mode:
    'depth' builds branches according to minimize depth in a shortest-path search
    between reference and child.
    'similarity' builds branches to minimize distance between parent and child.
    'linear' builds branches chronologically, extending forwards and backwards
    from each reference frame.

    :param container: variable container, needs to contain: dataset, channel,
    shape_t, t_annot
    :param sort_mode: method for building frame branches
    :param t_ignore: frames to ignore for analysis
    :param t_track: frames to track for analysis; supercedes t_ignore
    :param verbose: plot relevant scores for frame tree
    :return: container (updated entries for: t_list, p_list, r_list, s_list)
    """

    # pull variables from container
    dataset = container.get('dataset')
    channel = container.get('channel')
    shape_t = container.get('shape_t')
    t_annot = container.get('t_annot')

    if t_track is not None:
        t_ignore = np.setdiff1d(
            np.arange(shape_t),
            np.unique(list(t_annot.copy()) + list(t_track))
        )

    print('\nBuilding frame correlation graph...')
    d_full = get_all_pdists(dataset, shape_t, channel, pbar=True)
    if verbose:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(d_full)
        plt.show()
        fig.savefig(str(dataset / 'pw_distance.png'))
    
    if sort_mode == 'depth':
        print('\nSorting frames by depth along shortest path...')
        dist_sp, parents_sp = shortest_path(
            d_full,
            directed=False,
            return_predecessors=True,
            indices=t_annot
        )
        for i, t in enumerate(t_annot):
            parents_sp[i, t] = -1
        r_list = np.argmin(d_full[t_annot, :], axis=0)
        p_list = parents_sp[r_list, range(shape_t)]
        depths_sp = list(map(lambda x: get_depth(p_list, x), range(shape_t)))
        t_list = np.argsort(depths_sp)
    
    elif sort_mode == 'similarity':
        print('\nSorting frames by similarity to parent...')
        t_list = list(t_annot.copy())
        if t_ignore is not None:
            t_list = list(np.unique(t_list + list(t_ignore)))
        p_list = np.zeros((shape_t,))
        p_list[t_list] = -1

        d_temp = d_full.copy()
        np.fill_diagonal(d_temp, 2.1)  # dist_corrcoeff ranges from 0. to 2.0
        d_temp[:, t_list] = 2.1
        if t_ignore is not None:
            d_temp[list(t_ignore), :] = 2.1

        while len(t_list) < shape_t:
            parent, child = divmod(int(np.argmin(d_temp[t_list, :])), shape_t)
            t_list.append(child)
            p_list[child] = t_list[parent]
            d_temp[:, child] = 2.1
        t_list = np.array(t_list, dtype=int)
        p_list = np.array(p_list, dtype=int)
        r_list = np.argmin(d_full[t_annot, :], axis=0)

    else:
        print('\nSorting frames linearly off of reference frames...')
        temp = np.arange(shape_t)
        t_tree = [t_annot]
        p_tree = [[-1 for _ in t_annot]]
        for i in range(len(t_annot)):
            if t_annot[i] == 0:
                b = np.array([])
            elif i == 0:
                b = temp[:t_annot[i]]
            elif t_annot[i] == t_annot[i - 1] + 1:
                b = np.array([])
            else:
                b = temp[(t_annot[i] + t_annot[i - 1]) // 2:t_annot[i]]
            t_tree.append(b[::-1])
            p_tree.append(b[::-1] + 1)

            if t_annot[i] == shape_t - 1:
                f = np.array([])
            elif i == len(t_annot) - 1:
                f = temp[t_annot[i] + 1:]
            elif t_annot[i] == t_annot[i + 1] - 1:
                f = np.array([])
            else:
                f = temp[t_annot[i] + 1:(t_annot[i] + t_annot[i + 1]) // 2]
            t_tree.append(f)
            p_tree.append(f - 1)

        t_list = np.concatenate(t_tree).astype(int)
        p_list = np.concatenate(p_tree)[np.argsort(t_list)].astype(int)
        r_list = np.argmin(d_full[t_annot, :], axis=0)
        if t_ignore is not None:
            for t in list(t_ignore):
                if p_list[t] > t > 0:
                    p_list[t - 1] += 1
                elif p_list[t] < t < shape_t - 1:
                    p_list[t + 1] -= 1

    t_list = np.setdiff1d(t_list, t_annot, assume_unique=True)
    if t_ignore is not None:
        t_list = np.setdiff1d(t_list, np.array(t_ignore), assume_unique=True)

    s_list = get_undiscounted_scores_for_tree(d_full, p_list, shape_t)
    print(f'\nFrames sorted with max/mean distance'
          f'\t{np.max(s_list):.4f} / {np.mean(s_list):.4f}')

    if verbose:
        scores_r = get_undiscounted_scores_for_tree(
            d_full, list(t_annot[r] for r in r_list), shape_t
        )
        indicators = []
        for t in t_annot:
            indicators.append([t, 0, 0.5])
        plot_with_indicator_v(
            [scores_r],
            indicators,
            title=f'Scores for tree (to root) (m={np.mean(scores_r):.4f})'
        )
    
        scores_sp = get_undiscounted_scores_for_tree(d_full, p_list, shape_t)
        indicators = []
        for t in t_annot:
            indicators.append([t, 0, 0.5])
        plot_with_indicator_v(
            [scores_sp],
            indicators,
            title=f'Scores for tree (to parent) (m={np.mean(scores_sp):.4f})'
        )

    # push variables to container
    container.update({
        't_list': t_list,
        'p_list': p_list,
        'r_list': r_list,
        's_list': s_list,
    })

    return container
