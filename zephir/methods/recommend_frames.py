"""
recommend_frames.py: search and determine optimal reference frames to annotate for ZephIR.

Determine median frames to recommend as reference frames via k-medoids clustering
based on thumbnail distances (see build_pdists). Clustering is done iteratively,
such that one cluster is determined at a time. n_iter > 0 will re-iterate over
the existing list of median frames to fine-tune recommendations.

Usage:
    recommend_frames.py -h | --help
    recommend_frames.py -v | --version
    recommend_frames.py --dataset=<dataset> [options]

Options:
    -h --help                           	show this message and exit.
    -v --version                        	show version information and exit.
    --dataset=<dataset>  					path to data directory to analyze.
    --n_frames=<n_frames>  					number of reference frames to search for. [default: 5]
    --n_iter=<n_iter>  						number of iterations for optimizing results; -1 to uncap. [default: 0]
    --t_list=<t_list>  						frames to analyze.
    --channel=<channel>  					data channel to use for calculating correlation coefficients.
    --save_to_metadata=<save_to_metadata>  	save t_ref to metadata.json. [default: True]
    --verbose=<verbose>  					return score plots during search. [default: False]
"""

from collections import OrderedDict
from docopt import docopt

from ..__version__ import __version__
from ..methods.build_pdists import get_all_pdists
from ..utils.utils import *


def recommend_frames(
    dataset, n_frames, n_iter, t_list, channel,
    save_to_metadata, verbose
):

    metadata = get_metadata(dataset)
    shape_t = metadata['shape_t']
    if t_list is None:
        t_list = list(range(shape_t))

    print('Building frame correlation graph...')
    d_full = get_all_pdists(dataset, shape_t, channel, pbar=True)
    d_slice = (d_full[t_list, :])[:, t_list]

    scores = np.mean(d_slice, axis=-1)
    opt_score, med_idx = np.min(scores), np.argmin(scores)

    i_ref = [med_idx]
    t_ref = [t_list[med_idx]]
    s_ref = [opt_score]
    scores = d_slice[med_idx, :]
    pbar = tqdm(range(n_frames - 1), desc='Optimizing reference frames', unit='n_frames')
    for i in pbar:
        d_adj = np.append(
            d_slice.copy()[:, :, None],
            np.tile(scores[None, :, None], (d_slice.shape[0], 1, 1)),
            axis=-1
        )
        d_opt = np.min(d_adj, axis=-1)
        iscores = np.mean(d_opt, axis=-1)
        opt_score, new_midx = np.min(iscores), np.argmin(iscores)

        i_ref.append(new_midx)
        t_ref.append(t_list[new_midx])
        s_ref.append(opt_score)
        scores = d_opt[new_midx, :]
    print(f'\nFirst pass reference frames: {t_ref}'
          f'\nCurrent optimized score: {opt_score:.4f}')

    if verbose:
        plot_with_indicator_v(
            [scores],
            [[t, np.min(scores), 0.5] for t in t_ref],
            x_list=[t_list],
            title=f'First pass mean score: {opt_score:.4f}'
        )

    print(f'\nIterating over found reference frames...')
    n_i = 0
    while True:
        if 0 <= n_iter <= n_i:
            break
        kscore = opt_score
        for i in range(n_frames):
            i_ref_temp = i_ref.copy()
            i_ref_temp.pop(i)
            d_adj = d_slice.copy()[:, :, None]
            for t in i_ref_temp:
                d_adj = np.append(
                    d_adj,
                    np.tile(d_slice.copy()[t, None, :, None],
                            (d_adj.shape[0], 1, 1)),
                    axis=-1
                )

            iscores = np.mean(np.min(d_adj, axis=-1), axis=-1)
            jscore, new_midx = np.min(iscores), np.argmin(iscores)

            if jscore < kscore:
                i_ref[i] = new_midx
                t_ref[i] = t_list[new_midx]
                kscore = jscore

        if kscore < opt_score:
            opt_score = kscore
            n_i += 1
            if verbose:
                print(f'\nIter#{n_i}\tCurrent reference frames: {t_ref}'
                      f'\t\tCurrent optimal score: {opt_score:.4f}')
        else:
            break
    print(f'\nFinal reference frames: {t_ref}\n'
          f'Final optimized score: {opt_score:.4f}')

    if verbose:
        plot_with_indicator_v(
            [scores],
            [[t, np.min(scores), 0.5] for t in t_ref],
            x_list=[t_list],
            title=f'Final mean score: {opt_score:.4f}'
        )

        plt.figure()
        # plt.title('opt_score vs N_ref')
        plt.xlabel('Number of reference frames')
        plt.ylabel('Mean distance to reference')
        plt.ylim(0, np.max(s_ref)+0.05)
        plt.plot(np.arange(1, len(s_ref) + 1), s_ref)
        plt.show()

    if save_to_metadata:
        update_metadata(dataset, {f't_ref_fn{len(t_list)}': [int(i) for i in t_ref]})


def main():
    args = docopt(__doc__, version=f'ZephIR recommend_frames {__version__}')
    print(args, '\n')

    recommend_frames(
        dataset=Path(args['--dataset']),
        n_frames=int(args['--n_frames']),
        n_iter=int(args['--n_iter']),
        t_list=eval(args['--t_list']) if args['--t_list'] else None,
        channel=int(args['--channel']) if args['--channel'] else None,
        save_to_metadata=args['--save_to_metadata'] in ['True', 'Y', 'y'],
        verbose=args['--verbose'] in ['True', 'Y', 'y'],
    )


if __name__ == '__main__':
    main()
