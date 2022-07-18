"""
track_frame.py: track keypoints in a single frame.

Annotator GUI macro for ZephIR to track keypoints in a single frame (and its
child frames, if specified). Necessary components are loaded from checkpoint.pt
to expedite the process.
"""

import shutil
from docopt import docopt

from . import *
from ..models.container import Container
from ..utils.utils import *
from ..__version__ import __version__


def track_frame(
    dataset,
    annotation,
    t_idx,
    restrict_update=True,
    recompile_model=False,
    update_children=False):

    if len(load_checkpoint(dataset, verbose=True)) <= 1:

        if (dataset / 'args.json').is_file():
            with open(str(dataset / 'args.json')) as json_file:
                args = json.load(json_file)
        else:
            print('Creating checkpoint with default args...')
            from .. import main as zeir
            args = docopt(zeir.__doc__, argv=[f'--dataset={dataset}'])

        update_checkpoint(dataset, {'__version__': __version__, 'args': args})

        if torch.cuda.is_available():
            print('\n*** GPU available!')
            dev = 'cuda'
        # elif torch.backends.mps.is_available():
        #     dev = 'mps'
        else:
            dev = 'cpu'
        print(f'\nUsing device: {dev}\n\n')

        container = Container(
            dataset=dataset,
            allow_rotation=args['--allow_rotation'] in ['True', 'Y', 'y'],
            channel=int(args['--channel']) if args['--channel'] else None,
            dev=dev,
            exclude_self=args['--exclude_self'] in ['True', 'Y', 'y'],
            exclusive_prov=(bytes(args['--exclusive_prov'], 'utf-8')
                            if args['--exclusive_prov'] else None),
            gamma=float(args['--gamma']),
            include_all=args['--include_all'] in ['True', 'Y', 'y'],
            n_frame=int(args['--n_frame']),
            z_compensator=float(args['--z_compensator']),
        )
        recompile_model = True

    if not (dataset / 'backup').is_dir():
        Path.mkdir(dataset / 'backup')
    now = datetime.datetime.now()
    now_ = now.strftime("%m_%d_%Y_%H_%M_%S")
    shutil.copy(dataset / 'checkpoint.pt',
                dataset / 'backup' / f'checkpoint_{now_}.pt')

    print('\nFetching current results...\n')
    args = get_checkpoint(dataset, 'args')
    container = get_checkpoint(dataset, 'container')

    container.update({'exclude_self': False})
    container, _results = build_annotations(
        container=container,
        annotation=annotation,
        t_ref=None,
        wlid_ref=eval(args['--wlid_ref']) if args['--wlid_ref'] else None,
        n_ref=int(args['--n_ref']) if args['--n_ref'] else None,
    )
    container.update({'exclude_self': True})
    container, _ = build_annotations(
        container=container,
        annotation=annotation,
        t_ref=None,
        wlid_ref=eval(args['--wlid_ref']) if args['--wlid_ref'] else None,
        n_ref=int(args['--n_ref']) if args['--n_ref'] else None,
    )

    if recompile_model:
        container, zephir, zephod = build_models(
            container=container,
            dimmer_ratio=float(args['--dimmer_ratio']),
            grid_shape=(5, 2 * (int(args['--grid_shape']) // 2) + 1,
                        2 * (int(args['--grid_shape']) // 2) + 1),
            fovea_sigma=(1, float(args['--fovea_sigma']),
                         float(args['--fovea_sigma'])),
            n_chunks=int(args['--n_chunks']),
        )
        container = build_springs(
            container=container,
            load_nn=args['--load_nn'] in ['True', 'Y', 'y'],
            nn_max=int(args['--nn_max']),
        )
        container = build_tree(
            container=container,
            sort_mode=str(args['--sort_mode']),
            t_ignore=eval(args['--t_ignore']) if args['--t_ignore'] else None,
        )
    else:
        zephir = get_checkpoint(dataset, 'zephir')
        zephod = get_checkpoint(dataset, 'zephod')

    t_list = [t_idx]
    if update_children:
        p_list = container.get('p_list')
        parents, p = [t_idx], 0
        while p < len(parents):
            children = np.where(np.array(p_list) == parents[p])[0]
            for c in children:
                t_list.append(c)
                parents.append(c)
            p += 1
    t_list = np.array(t_list, dtype=int)

    container, results = track_all(
        container=container,
        results=_results,
        zephir=zephir,
        zephod=zephod,
        clip_grad=float(args['--clip_grad']),
        lambda_t=float(args['--lambda_t']),
        lambda_d=float(args['--lambda_d']),
        lambda_n=float(args['--lambda_n']),
        lambda_n_mode=args['--lambda_n_mode'],
        lr_ceiling=float(args['--lr_ceiling']),
        lr_coef=float(args['--lr_coef']),
        lr_floor=float(args['--lr_floor']),
        motion_predict=args['--motion_predict'] in ['True', 'Y', 'y'],
        n_epoch=int(args['--n_epoch']),
        n_epoch_d=(int(args['--n_epoch_d'])
                   if float(args['--lambda_d']) > 0 else 0),
        restrict_update=restrict_update,
        _t_list=t_list,
    )

    return (
        list(t_list),
        results[t_list, ...],
        list(container.get('worldline_id')),
        container.get('provenance')[t_list, ...]
    )
