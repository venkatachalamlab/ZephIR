"""
overwrite_checkpoint.py: overwrite existing key and value in ZephIR checkpoint.

Usage:
	overwrite_checkpoint.py -h | --help
	overwrite_checkpoint.py -v | --version
	overwrite_checkpoint.py --dataset=<dataset> --key=<key> --value=<value> [options]

Options:
	-h --help                           	show this message and exit.
	-v --version                        	show version information and exit.
	--dataset=<dataset>  					path to data directory to analyze.
	--key=<key>  							name of item in checkpoint to overwrite.
	--value=<value>  						new item to write into checkpoint.
"""

from docopt import docopt

from ..__version__ import __version__

from . import *
from ..models.container import Container
from ..utils.utils import *


def overwrite_checkpoint(dataset, key, value):
	_checkpoint = load_checkpoint(dataset, fallback=False, verbose=True)

	updated_items = 0

	if key == 'args' and value in ['all', '', None]:
		if (dataset / 'args.json').is_file():
			with open(str(dataset / 'args.json')) as json_file:
				args = json.load(json_file)
		update_checkpoint(dataset, {'args': args})
		updated_items += 1

	if key in _checkpoint:
		print('Updating checkpoint main dict...')
		update_checkpoint(dataset, {key: type(_checkpoint[key])(value)})
		updated_items += 1

	if key in _checkpoint['args'] or f'--{key}' in _checkpoint['args']:
		print('Updating saved args...')
		args = _checkpoint['args']
		args[f'--{key}'] = type(args[f'--{key}'])(value)
		update_checkpoint(dataset, {'args': args})
		updated_items += 1

		_args_list = [
			'allow_rotation', 'channel', 'exclude_self', 'exclusive_prov',
			'gamma', 'include_all', 'n_frame', 'z_compensator'
		]
		if key in _args_list:
			print('Rebuilding container from updated args...')
			container = Container(
				dataset=dataset,
				allow_rotation=args['--allow_rotation'] in ['True', 'Y', 'y'],
				channel=int(args['--channel']) if args['--channel'] else None,
				dev='cpu',
				exclude_self=args['--exclude_self'] in ['True', 'Y', 'y'],
				exclusive_prov=(bytes(args['--exclusive_prov'], 'utf-8')
								if args['--exclusive_prov'] else None),
				gamma=float(args['--gamma']),
				include_all=args['--include_all'] in ['True', 'Y', 'y'],
				n_frame=int(args['--n_frame']),
				z_compensator=float(args['--z_compensator']),
			)
			updated_items += 1
		else:
			container = _checkpoint['container']
		_args_list = _args_list + ['t_ref', 'wlid_ref', 'n_ref']
		if key in _args_list:
			build_annotations(
				container=container,
				annotation=None,
				t_ref=eval(args['--t_ref']) if args['--t_ref'] else None,
				wlid_ref=eval(args['--wlid_ref']) if args['--wlid_ref'] else None,
				n_ref=int(args['--n_ref']) if args['--n_ref'] else None,
			)
			updated_items += 1
		_args_list = _args_list + ['dimmer_ratio', 'grid_shape', 'fovea_sigma', 'n_chunks']
		if key in _args_list:
			build_models(
				container=container,
				dimmer_ratio=float(args['--dimmer_ratio']),
				grid_shape=(5, 2 * (int(args['--grid_shape']) // 2) + 1,
							2 * (int(args['--grid_shape']) // 2) + 1),
				fovea_sigma=(1, float(args['--fovea_sigma']),
							 float(args['--fovea_sigma'])),
				n_chunks=int(args['--n_chunks']),
			)
			updated_items += 1
		_args_list = _args_list + ['load_nn', 'nn_max']
		if key in _args_list:
			build_springs(
				container=container,
				load_nn=args['--load_nn'] in ['True', 'Y', 'y'],
				nn_max=int(args['--nn_max']),
			)
			updated_items += 1
		_args_list = _args_list + ['sort_mode', 't_ignore']
		if key in _args_list:
			build_tree(
				container=container,
				sort_mode=str(args['--sort_mode']),
				t_ignore=eval(args['--t_ignore']) if args['--t_ignore'] else None,
			)
			updated_items += 1

	if updated_items == 0:
		print('*** KEY NOT FOUND! Check arguments and try again.')

	return


def main():
	args = docopt(__doc__, version=f'ZephIR overwrite_checkpoint {__version__}')
	print(args, '\n')

	overwrite_checkpoint(
		dataset=Path(args['--dataset']),
		key=str(args['--key']),
		value=eval(args['--value']),
	)


if __name__ == '__main__':
	main()
