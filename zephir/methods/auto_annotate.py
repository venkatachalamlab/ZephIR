"""
auto_annotate.py: use ZephOD's feature detection to automatically annotate a frame in a dataset.

Usage:
    auto_annotate -h | --help
    auto_annotate -v | --version
    auto_annotate --dataset=<dataset> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset=<dataset>                 path to data directory to analyze.
    --tidx=<tidx>                       frame number or time index to analyze. [default: 0]
    --channel=<channel>                 data channel to register.
    --cuda=<cuda>                       check if a CUDA-compatible GPU is available for  use. [default: True]
    --min_distance=<min_distance>       minimum distance between detected features in pixels. [default: 4]
    --min_val=<min_val>                 minimum intensity value for a peak. [default: 1]
    --model=<model>                     path to checkpoint for model.
"""


from docopt import docopt
import shutil
from skimage.feature import peak_local_max

from ..__version__ import __version__
from ..utils.utils import *
from ..zephod.model import ZephOD


def auto_annotate(dataset: Path, tidx: int, args: dict):

    if torch.cuda.is_available() and args['--cuda'] in ['True', 'Y', 'y']:
        # Moving to GPU
        print('\n*** GPU available!')
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    if args['--model'] is None:
        checkpoint_path = Path(__file__).parent.parent / 'zephod' / 'model.pt'
    else:
        checkpoint_path = Path(args['--model'])
    if Path.is_file(checkpoint_path):
        print('\nPrevious checkpoint available.')
        try:
            checkpoint = torch.load(checkpoint_path)
        except RuntimeError:
            print('*** DEVICE INCOMPATIBLE! Mapping CUDA tensors to CPU...')
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        print('\n******* ERROR: model not found! Exiting...')
        exit()

    # loading other user arguments
    channel = int(args['--channel']) if args['--channel'] else None
    min_distance = int(args['--min_distance'])

    # performing inference with ZephOD
    # this produces a probability map of the desired features
    print('\nRunning feature detection with ZephOD...')
    model = ZephOD(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(dev)
    model.eval()
    vol = get_data(dataset, t=tidx, c=channel)
    pred = model(vol)
    im = to_numpy(pred[0, 0] * 255)
    peaks = peak_local_max(im, min_distance=min_distance)

    # removing peaks at which the pixel intensity does not meet a minimum
    low_peak_list = []
    for i, (z, y, x) in enumerate(peaks):
        if np.max(vol[:, z, y, x]) < float(args['--min_val'])/255:
            low_peak_list.append(i)
    peaks = np.delete(peaks, low_peak_list, axis=0)

    print(f'*** Found {peaks.shape[0]} keypoints!')

    # compiling results into annotator-ready format
    shape_n = peaks.shape[0]
    img_shape = np.array(im.shape)
    provenance = np.array([b'ZEIR'] * shape_n)
    xyz_pd = np.concatenate(
        (tidx * np.ones((shape_n, 1)),
         peaks[:, ::-1] / img_shape[None, ::-1],
         np.arange(shape_n)[:, None],
         -1 * np.ones((shape_n, 1)),
         provenance[:, None]),
        axis=-1
    )

    # creating backup
    if not (dataset / 'backup').is_dir():
        Path.mkdir(dataset / 'backup')
    if (dataset / 'annotations.h5').is_file():
        print('\n\n*** WARNING: Existing annotations file found!\n'
              'This process will create a backup of the existing file and '
              'overwrite any annotations in the selected frame with new auto-detected annotations.\n'
              'Please note that the new annotations are NOT properly linked or '
              'assigned to any existing worldlines.')
        now = datetime.datetime.now()
        now_ = now.strftime("%m_%d_%Y_%H_%M_%S")
        shutil.copy(dataset / 'annotations.h5',
                    dataset / 'backup' / f'annotations_{now_}.h5')

        # adding any existing annotations to the new annotation set
        # this skips the selected frame, essentially
        annotation = get_annotation_df(dataset)
        for t in np.unique(annotation['t_idx']):
            if t == tidx:
                continue
            u, annot, prov = get_annotation(annotation, t, None, False)
            xyz_pd = np.append(
                xyz_pd,
                np.concatenate(
                    (np.ones((len(u), 1)) * t,
                     annot / 2 + 0.5,
                     u[:, None],
                     -1 * np.ones((len(u), 1)),
                     prov[:, None]),
                    axis=-1
                ),
                axis=0
            )

    # writing results to file
    columns = {
        't_idx': np.uint32,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'worldline_id': np.uint32,
        'parent_id': np.uint32,
        'provenance': np.dtype("S4"),
    }
    f = h5py.File(dataset / 'annotations.h5', mode='w')
    data = np.array(list(range(1, xyz_pd.shape[0] + 1)), dtype=np.uint32)
    f.create_dataset('id', shape=(xyz_pd.shape[0],), dtype=np.uint32, data=data)
    for i, c in enumerate(columns.keys()):
        if c == 'provenance':
            data = np.array(xyz_pd[:, i], dtype=columns[c])
        else:
            data = np.array(xyz_pd[:, i].astype(np.float32), dtype=columns[c])
        f.create_dataset(c, shape=(xyz_pd.shape[0],), dtype=columns[c], data=data)
    f.close()

    print('\n\n*** DONE!')
    return


def main():
    args = docopt(__doc__, version=f'ZephIR auto_annotate {__version__}')
    # print(args, '\n')

    dataset = Path(args['--dataset'])

    auto_annotate(
        dataset=dataset,
        tidx=int(args['--tidx']),
        args=args
    )


if __name__ == '__main__':
    main()
