"""
extract_traces.py: Extract fluorescent activity traces.

Using keypoint coordinates in annotations.h5, both manual annotations and Zephir
results, extracts fluorescent activity traces by analyzing pixel intensities.
This uses build arguments and container attributes from checkpoint.pt.
A modified version of the Zephir model is used to grid sample pixels around each
annotation and exclude pixels from neighboring cells.

Adapted from program by Mahdi Torkashvand

Usage:
    extract_traces -h | --help
    extract_traces -v | --version
    extract_traces --dataset=<dataset> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset=<dataset>                 path to data directory to analyze.
    --channel=<channel>                 data channel to pull traces from.
    --cuda=<cuda>  						check if a CUDA-compatible GPU is available for  use. [default: True]
    --cutoff=<cutoff>  					cutoff frequency for lowpass filter. [default: 1.0]
    --debleach=<debleach>  				whether to apply a debleaching step. [default: True]
    --dist_thresh=<dist_thresh>  		threshold for pw-frame distance to exclude. [default: 2.1]
    --n_chunks=<n_chunks>               number of steps to divide the forward pass into. [default: 10]
    --n_cluster=<n_cluster>     		threshold for clustering for heatmap. [default: 1]
    --nn_max=<nn_max>  					maximum number of neighbors to mask. [default: 5]
    --rma_channel=<rma_channel>  		data channel for removing multiplicative artifacts.
    --t_list=<t_list>  					time points to analyze.
    --wlid_ref=<wlid_ref>               subset of neurons to extract by worldline_id.
"""

from docopt import docopt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from skimage.transform import resize
from sklearn.cluster import SpectralClustering
import sklearn.neighbors as sk

from ..__version__ import __version__
from . import *
from ..models.zephir import ZephIR
from ..utils.utils import *


def double_exp(t, a1, b1, a2, b2):
    return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)


def extract_traces(
    dataset,
    cuda,
    channel,
    cutoff,
    debleach,
    n_chunks,
    n_cluster,
    nn_max,
    rma_channel,
    t_list,
    dist_thresh,
    wlid_ref,
    a_scalar=0.8,
    crop_shape=(3, 9, 9),
    eps=0.01,
    npx_to_keep=16,
    save_as_npy=True,
    save_as_fig=True,
    verbose=False):

    # checking for available CUDA GPU
    if cuda in ['True', 'Y', 'y'] and torch.cuda.is_available():
        print('\n*** GPU available!')
        dev = 'cuda'
    else:
        dev = 'cpu'
    print(f'\nUsing device: {dev}\n\n')

    container = get_checkpoint(dataset, 'container', verbose=True)
    container.update({
        'dev': dev,
        'exclude_self': False,
        'exclusive_prov': None,
    })
    container, results = build_annotations(
        container=container,
        annotation=None,
        t_ref=None,
        wlid_ref=wlid_ref,
        n_ref=None,
    )

    s_list = container.get('s_list')
    worldline_id = container.get('worldline_id')

    if t_list is None:
        t_list = list(container.get('t_annot'))

    # import data here
    print('Loading file...')
    metadata = get_metadata(dataset)
    img_shape = (metadata['shape_z'], metadata['shape_y'], metadata['shape_x'])
    shape_t = metadata['shape_t']
    shape_c = metadata['shape_c']
    n_neuron = results.shape[1]

    crop_spacing = tuple(np.array(crop_shape) / np.array(img_shape))
    nn_max = min(nn_max, n_neuron - 1)

    print(f'\nCompiling model...')
    model_kwargs = {
        'allow_rotation': False,
        'dimmer_ratio': 1.0,
        'fovea_sigma': (-1, -1, -1),
        'grid_shape': crop_shape,
        'grid_spacing': crop_spacing,
        'n_chunks': n_chunks,
        'n_frame': 1,
        'shape_n': n_neuron,
        'ftr_ratio': 1.0,
        'ret_stride': 1.0,
    }
    model = ZephIR(**model_kwargs).to(dev)

    if dist_thresh < 2.0:
        t_ignore = np.where(s_list > dist_thresh)[0]
        print(f'*** Excluding traces for frames: {list(t_ignore)}')
        t_list = np.setdiff1d(t_list, t_ignore)

    traces = np.empty((shape_c, n_neuron, shape_t)) * np.nan
    for t in tqdm(t_list, desc='Compiling raw traces', unit='frames'):
        data = get_slice(dataset, t).astype(float)
        with torch.no_grad():
            vol = to_tensor(data, n_dim=5, grad=False, dev=dev)
            model.theta.zero_()
            model.rho.zero_()
            model.rho.add_(to_tensor(results[t, ...], dev=dev).unsqueeze(0))
            descriptors = model(vol.unsqueeze(0))[0]
            descriptors = descriptors.view(n_neuron, shape_c, -1)
            grids = (model.grid + model.rho.view((-1, 1, 1, 1, 3))).view(n_neuron, -1, 3)
            grids *= to_tensor(np.array(img_shape)[::-1] / np.max(img_shape), n_dim=3, dev=dev)

            tree = sk.KDTree(results[t, ...])
            ind = (tree.query(results[t, ...], k=nn_max + 1, return_distance=False))[:, 1:]
            for n in range(n_neuron):
                mask = descriptors[n, ...].int()
                if nn_max > 0:
                    nn_idx = ind[n, :]
                    cdists = torch.cdist(
                        grids[n].unsqueeze(0).expand((nn_max, -1, -1)),
                        torch.stack([grids[n_] for n_ in nn_idx], dim=0)
                    )
                    for d in cdists:
                        mask[:, torch.mean(d, dim=-1) < eps] *= 0
                for c in range(shape_c):
                    nz = to_numpy(mask[c][torch.nonzero(mask[c], as_tuple=True)]).flatten()
                    if len(nz) == 0:
                        traces[c, n, t] = np.nan
                    else:
                        traces[c, n, t] = np.mean(np.sort(nz)[-npx_to_keep:]) / 255
    if save_as_npy:
        np.save(str(dataset / f'traces_raw.npy'), traces)

    for n in range(n_neuron):
        for c in range(shape_c):
            traces[c, n, t_list] = fill_nans(traces[c, n, t_list])

    # remove multiplicative artifact
    if rma_channel is not None:
        for n in tqdm(range(n_neuron), desc='Removing artifacts', unit='items'):
            params_r = curve_fit(
                double_exp, np.arange(len(t_list)),
                traces[rma_channel, n, t_list],
                bounds=(0., np.inf), method='trf'
            )[0]
            curve_r = double_exp(np.arange(len(t_list)), *params_r)
            if verbose:
                # plt.plot(t_list, traces[0, n, t_list], 'r-')
                plt.plot(t_list, traces[1, n, t_list], 'g-')
                # plt.plot(t_list, curve_r, 'r-', alpha=0.8)
                plt.show()

            artifacts = (traces[rma_channel, n, t_list] - curve_r) / curve_r
            traces[rma_channel, n, t_list] /= 1 + artifacts
            if channel is not None:
                traces[channel, n, t_list] /= 1 + a_scalar * artifacts
            else:
                for c in range(traces.shape[0]):
                    if c != rma_channel:
                        traces[c, n, t_list] /= 1 + a_scalar * artifacts
            for c in range(traces.shape[0]):
                traces[c, n, t_list] = fill_nans(traces[c, n, t_list])

    if channel is not None:
        traces = traces[None, channel, ...]

    # lowpass filter
    if cutoff > 0:
        for n in tqdm(range(n_neuron), desc='Applying lowpass filter', unit='items'):
            params = butter(N=2, Wn=cutoff, output='ba', btype='lowpass', fs=4.0, analog=False)
            for c in range(traces.shape[0]):
                traces[c, n, t_list] = filtfilt(
                    params[0], params[1], traces[c, n, t_list]
                )
                traces[c, n, t_list] = fill_nans(traces[c, n, t_list])

            if verbose:
                plt.plot(t_list, traces[1, n, t_list], 'g-')
                plt.show()

    # debleaching
    if debleach:
        for n in tqdm(range(n_neuron), desc='Debleaching', unit='items'):
            for c in range(traces.shape[0]):
                params = curve_fit(
                    double_exp,
                    np.arange(len(t_list)),
                    traces[c, n, t_list],
                    bounds=(0., np.inf),
                    method='trf'
                )[0]
                curve = double_exp(np.arange(len(t_list)), *params)
                curve = np.clip(curve, np.nanmin(traces[c, n, t_list]) + 0.002, 1.0)
                if verbose:
                    plt.plot(t_list, traces[c, n, t_list], 'g-')
                    plt.plot(t_list, curve, 'g-', alpha=0.8)
                    plt.show()
                traces[c, n, t_list] *= curve[0, None] / curve
                traces[c, n, t_list] = fill_nans(traces[c, n, t_list])

    traces = (traces - traces[:, :, t_list[0], None]) / traces[:, :, t_list[0], None]
    for n in range(n_neuron):
        traces[0, n, :] = fill_nans(traces[0, n, :])

    if save_as_npy:
        np.save(str(dataset / f'traces.npy'), traces)

    if save_as_fig:
        # plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(5, n_neuron))
        colors = plt.cm.hsv(np.linspace(0, 1, n_neuron + 1)[:-1])[:, :3]
        offset = np.cumsum(np.append(0.0,
                                     np.nanmax(traces[0, :-1, :], axis=-1)
                                     - np.nanmin(traces[0, :-1, :], axis=-1)
                                     ))
        s = traces[0] - np.nanmin(traces[0], axis=-1)[:, None] + offset[:, None]
        for n in range(n_neuron):
            plt.plot(s[n], linewidth=1, color=colors[n])
        plt.yticks(s[:, 0], [str(n) for n in worldline_id])
        plt.ylim(np.nanmin(s) - 0.01, np.nanmax(s) + 0.01)
        plt.xlabel('Time (frames)')
        plt.ylabel('Neuron ID')
        plt.savefig(str(dataset / f'traces.png'), format='png')
        plt.close('all')

        plt.figure(figsize=(10, 5))
        s = (traces[0] - np.nanmin(traces[0], axis=-1)[:, None])
        s = s / np.clip(np.nanmax(s, axis=-1)[:, None], 3.0, 12.0)
        for n in range(n_neuron):
            s[n, :] = fill_nans(s[n, :])
        print(np.mean(s), np.min(s), np.max(s))
        sm = np.zeros((n_neuron, n_neuron))
        for i in range(n_neuron):
            for j in range(i, n_neuron):
                sm[i, j] = np.corrcoef(s[i, :], s[j, :])[0, 1] + 1
        sm = sm + np.transpose(sm)
        sc = SpectralClustering(n_cluster, affinity='precomputed')
        fc = sc.fit_predict(sm)
        hm = s[np.argsort(fc), :]
        # plt.imshow(hm)
        plt.imshow(resize(hm, (hm.shape[0], 2 * hm.shape[0])))
        plt.yticks([])
        plt.xlabel('Time (frames)')
        plt.ylabel('Neuron ID')
        plt.tight_layout()
        plt.savefig(str(dataset / f'traces_hm.png'), format='png', dpi=500)
        plt.close('all')

    return traces


def main():
    args = docopt(__doc__, version=f'Zephir extract_traces {__version__}')
    print(args, '\n')

    extract_traces(
        dataset=Path(args['--dataset']),
        channel=int(args['--channel']) if args['--channel'] else None,
        cuda=args['--cuda'] in ['True', 'Y', 'y'],
        cutoff=float(args['--cutoff']),
        debleach=args['--debleach'] in ['True', 'Y', 'y'],
        n_chunks=int(args['--n_chunks']),
        n_cluster=int(args['--n_cluster']),
        nn_max=int(args['--nn_max']),
        rma_channel=int(args['--rma_channel']) if args['--rma_channel'] else None,
        t_list=list(args['--t_list']) if args['--t_list'] else None,
        dist_thresh=float(args['--dist_thresh']),
        wlid_ref=eval(args['--wlid_ref']) if args['--wlid_ref'] else None,
    )


if __name__ == '__main__':
    main()
