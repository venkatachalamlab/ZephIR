"""
ZephOD feature detection.

Usage:
    zephod -h | --help
    zephod -v | --version
    zephod --dataset=<dataset> --model=<model> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset=<dataset>                 path to data directory to analyze.
    --model=<model>                     path to checkpoint for model.
    --channel=<channel>                 data channel to use as input.
    --cuda=<cuda>                       check if a CUDA-compatible GPU is available for  use. [default: True]
"""

from docopt import docopt

from ..__version__ import __version__
from ..utils.utils import *
from .model import ZephOD


def run_zephod(
        dataset=Path('.'),
        dev=torch.device('cpu'),
        channel=None,
        model_kwargs=None,
        state_dict=None):

    model = ZephOD(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    vol = get_slice(dataset, 0)
    data_file = h5py.File(dataset / 'centers.h5', 'w')
    data = data_file.create_dataset(
        'data', (0, *vol.shape[1:]),
        chunks=(1, *vol.shape[1:]),
        dtype=np.uint8,
        compression='gzip',
        compression_opts=5,
        maxshape=(None, *vol.shape[1:])
    )
    times = data_file.create_dataset(
        'times', (0,),
        chunks=(1,),
        dtype=np.float64,
        maxshape=(None,),
    )
    rec = cv2.VideoWriter(
        str(dataset / 'centers.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), 30,
        (vol.shape[3], vol.shape[2]),
        False
    )

    t_list = get_times(dataset)
    tpbar = tqdm(t_list, desc='Analyzing movie', unit='frames')
    for t in tpbar:
        vol = get_slice(dataset, t)
        if channel is not None:
            vol = vol[channel, np.newaxis, ...]
        pred = model(vol)

        data.resize((t + 1, *vol.shape[1:]))
        data[t, ...] = to_numpy(pred[0, 0])
        times.resize((t + 1,))
        times[t] = t
        rec.write(
            np.max(
                np.transpose(to_numpy(pred[0, 0] * 255).astype(np.uint8)),
                axis=0
            )
        )
    data_file.close()
    rec.release()

    print('\n\n*** DONE!')
    return


def main():
    args = docopt(__doc__, version=f'ZephOD {__version__}')
    print(args, '\n')

    if torch.cuda.is_available() and args['--cuda'] in ['True', 'Y', 'y']:
        # Moving to GPU
        print('\n*** GPU available!')
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    if args['--model'] is None:
        checkpoint_path = Path(__file__).parent / 'model.pt'
    else:
        checkpoint_path = Path(__file__).parent / 'models' / Path(args['--model'] + '.pt')
    if Path.is_file(checkpoint_path):
        print('\nPrevious checkpoint available.')
        try:
            checkpoint = torch.load(checkpoint_path)
        except RuntimeError:
            print('*** CUDA NOT AVAILABLE! Mapping CUDA tensors to CPU...')
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        print('\n******* ERROR: model not found! Exiting...')
        exit()

    run_zephod(
        dataset=Path(args['--dataset']),
        dev=dev,
        channel=int(args['--channel']) if args['--channel'] else None,
        model_kwargs=checkpoint['model_kwargs'],
        state_dict=checkpoint['state_dict'],
    )


if __name__ == '__main__':
    main()
