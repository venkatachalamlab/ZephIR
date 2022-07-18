"""
Training ZephOD feature detection network

Usage:
    train_zephod -h | --help
    train_zephod -v | --version
    train_zephod --dataset=<dataset> --model=<model> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset=<dataset>                 path to data directory to analyze.
    --batch_size=<batch_size>           batch size in volumes. [default: 1]
    --channel=<channel>                 data channel to use as input.
    --cuda=<cuda>                       check if a CUDA-compatible GPU is available for  use. [default: True]
    --loss_func=<loss_func>             loss function to use. [default: BCE]
    --lr_init=<lr_init>                 learning rate for optimizer. [default: 1.0]
    --n_epoch=<n_epoch>                 number of epochs to train for. [default: 64]
    --n_trn=<n_trn>                     number of training samples per epoch. [default: 50]
    --n_val=<n_val>                     number of validation samples per epoch. [default: 5]
    --override=<override>               override existing checkpoint. [default: False]
"""

from docopt import docopt
import random
import shutil
import torch.nn as nn
import torch.optim as optim

from ..__version__ import __version__
from ..utils.getters import *
from zephir.models.losses import corr_loss
from ..utils.utils import *
from .augment import generate_synthetic_data, identify
from .model import ZephOD


def train_model(
        dataset=Path('.'),
        dev=torch.device('cpu'),
        channel=None,
        n_epoch=10,
        loss_func='BCE',
        lr_init=0.1,
        batch_size=1,
        n_trn=100,
        n_val=10,
        model_kwargs=None,
        state_dict=None):

    annotation = get_annotation_df(dataset)
    t_note = np.unique(annotation['t_idx']).astype(int)
    t_note = np.sort(t_note)
    nn_list = [len(get_annotation(annotation, t)[0]) for t in t_note]
    n_neuron, t_neuron = np.max(nn_list), t_note[np.argmax(nn_list)]
    worldline_id, _, _ = get_annotation(annotation, t_neuron)
    for t in t_note:
        u, _, _ = get_annotation(annotation, t)
        if len(u) == n_neuron:
            worldline_id = u
            t_neuron = t
            break
    print(f'Using frame #{t_neuron} as initial reference with {n_neuron} annotations found...')

    xyz_note = []
    for t in t_note:
        u, annot, prov = get_annotation(annotation, t)
        u_idx = np.array([np.where(u == w)[0][-1] for w in worldline_id if w in u], dtype=int)
        annot = annot[u_idx, ...]
        if annot.shape[0] > n_neuron or annot.shape[0] == 0:
            t_note = np.setdiff1d(t_note, [t])
            continue
        elif annot.shape[0] < n_neuron:
            t_note = np.setdiff1d(t_note, [t])
            continue
        xyz_note.append(annot)
    print(f'\nAnnotations loaded for frames {t_note} with shape: {np.array(xyz_note).shape}')
    annots = np.array(xyz_note)

    all_isolates = []
    pbar = tqdm(
        range(len(t_note)),
        desc='Finding isolated features',
        unit='frames',
        postfix=f'Features={len(all_isolates)}'
    )
    for i in pbar:
        # for i in range(len(t_note)):
        data = get_slice(dataset, t_note[i])
        if channel is not None:
            data = data[channel, np.newaxis, ...]
        isolated_neurons = identify(data, annots[i], 6)
        for n in isolated_neurons:
            all_isolates.append(n)
        pbar.set_postfix(Neurons=f'{len(all_isolates)}')
    # print(f'\n\nFound isolated neurons: {len(all_isolates)}')

    print('\nCompiling model...')
    vol = get_slice(dataset, t_note[0])
    if channel is not None:
        vol = vol[channel, np.newaxis, ...]
    if model_kwargs is None:
        model_kwargs = {
            'img_shape': vol.shape[1:],
            'n_channels_in': 5 * vol.shape[0],
            'n_channels_out': 1,
            'init_nodes': 16,
            'kernel': (1, 3, 3),
            'padding': 1,
            'pool_kernel': (2, 2, 2)
        }
    model = ZephOD(**model_kwargs).to(dev)
    if state_dict is not None:
        print('Loading model from existing state_dict...')
        model.load_state_dict(state_dict)
    optimizer = optim.Adadelta(model.parameters(), lr=lr_init)
    if loss_func == 'BCE':
        loss_function = nn.BCELoss()
    else:
        loss_function = corr_loss

    n_val = max(min(n_val, len(t_note) - 1), 0)
    t_list = np.array(t_note)[:-n_val]
    t_val = np.array(t_note)[-n_val:]

    print('\n\n\n******* BEGIN TRAINING *******\n')
    loss_list = []
    pbar = tqdm(range(n_epoch), desc='Training model', unit='epochs')
    for epoch in pbar:
        # for epoch in range(n_epoch):
        model.train()
        tpbar = tqdm(range(n_trn), leave=False, desc='Training samples', unit='vol')
        for _ in tpbar:
            # for i in range(n_trn):
            with torch.no_grad():
                X, Y = [], []
                for j in range(batch_size):
                    t_idx = random.randint(0, len(t_list) - 1)
                    synth, labels = generate_synthetic_data(
                        vol, annots[t_idx], all_isolates, True
                    )
                    X.append(to_tensor(synth, dev=dev))
                    Y.append(to_tensor(labels, dev=dev))
                X_trn, Y_trn = torch.stack(X, dim=0), torch.stack(Y, dim=0)
            optimizer.zero_grad()
            pred = model(X_trn)
            loss = loss_function(pred, Y_trn)
            loss.backward()
            with torch.no_grad():
                tpbar.set_postfix(t=f'{t_list[t_idx]}', Loss=f'{loss.item():.4f}')
        optimizer.step()

        with torch.no_grad():
            if n_val > 0:
                model.eval()
                X_val, Y_val = [], []
                for i in tqdm(range(len(t_val)), leave=False, desc='Validating samples', unit='vol'):
                    # for i in range(len(t_val)):
                    synth, labels = generate_synthetic_data(
                        vol, annots[-n_val + i], all_isolates, True
                    )
                    X_val.append(to_tensor(synth, dev=dev))
                    Y_val.append(to_tensor(labels, dev=dev))
                X_val, Y_val = torch.stack(X_val, dim=0), torch.stack(Y_val, dim=0)
                val_pred = model(X_val)
                val_loss = loss_function(val_pred, Y_val)
            else:
                val_loss = 0.
        pbar.set_postfix(Loss=f'{loss.item():.4f}', Validation_loss=f'{val_loss:.4f}')

        with torch.no_grad():
            print(f'Epoch: {epoch + 1:3} / {n_epoch}'
                  f'\t\tLoss: {loss:.4f}\t\tVal loss: {val_loss:.4f}')
        loss_list.append(loss.item())

    checkpoint = {
        'model_kwargs': model_kwargs,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'loss_list': loss_list
    }

    with torch.no_grad():
        X_val = Y_val = val_pred = None
        if n_val > 0:
            model.eval()
            X_val, Y_val = [], []
            for i in tqdm(range(len(t_val)), leave=False, desc='Validating samples', unit='vol'):
                # for i in range(len(t_val)):
                synth, labels = generate_synthetic_data(
                    vol, annots[-n_val + i], all_isolates, True
                )
                X_val.append(to_tensor(synth, dev=dev))
                Y_val.append(to_tensor(labels, dev=dev))
            X_val, Y_val = torch.stack(X_val, dim=0), torch.stack(Y_val, dim=0)
            val_pred = model(X_val)
    return checkpoint, (X_val, Y_val, val_pred), (to_numpy(model.alpha), to_numpy(model.gamma))


def main():
    args = docopt(__doc__, version=f'ZephOD {__version__}')
    print(args, '\n')

    if torch.cuda.is_available() and args['--cuda'] in ['True', 'Y', 'y']:
        # Moving to GPU
        print('\n*** GPU available!\n')
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    checkpoint_path = Path(args['--dataset']) / 'zephod.pt'
    override = args['--override'] in ['True', 'Y', 'y']
    checkpoint = {
        'model_kwargs': None,
        'state_dict': None,
        'opt_dict': None,
        'loss_list': None
    }
    if Path.is_file(checkpoint_path):
        print('\nPrevious checkpoint available.')
        if not (checkpoint_path.parent / 'bin').is_dir():
            Path.mkdir(checkpoint_path.parent / 'bin')
        now = datetime.datetime.now()
        now_ = now.strftime("%m_%d_%Y_%H_%M_%S")
        shutil.copy(
            checkpoint_path,
            checkpoint_path.parent / 'backup' / f'zephod_{now_}.pt'
        )
        if not override:
            try:
                checkpoint = torch.load(checkpoint_path)
            except RuntimeError:
                print('*** CUDA NOT AVAILABLE! Mapping CUDA tensors to CPU...')
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    checkpoint, validation, params = train_model(
        dataset=Path(args['--dataset']),
        dev=dev,
        channel=int(args['--channel']) if args['--channel'] else None,
        n_epoch=int(args['--n_epoch']),
        loss_func=str(args['--loss_func']),
        lr_init=float(args['--lr_init']),
        batch_size=int(args['--batch_size']),
        n_trn=int(args['--n_trn']),
        n_val=int(args['--n_val']),
        model_kwargs=checkpoint['model_kwargs'],
        state_dict=checkpoint['state_dict'],
    )
    print(f'\nSaving checkpoint to: {checkpoint_path.as_posix()}')
    torch.save(checkpoint, checkpoint_path)

    X_val, Y_val, val_pred = validation
    if val_pred is not None:
        print('\nSaving prediction to a new file...')
        print(X_val.shape, val_pred.shape, Y_val.shape)
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        axes[0].imshow(np.max(np.max(to_numpy(X_val[0]), axis=0), axis=0), vmin=0, vmax=1)
        axes[0].set_title('Input Volume')
        axes[1].imshow(np.max(to_numpy(val_pred[0][0]), axis=0), vmin=0, vmax=1)
        axes[1].set_title('Output Labels')
        im = axes[2].imshow(np.max(to_numpy(Y_val[0][0]), axis=0), vmin=0, vmax=1)
        axes[2].set_title('GT Labels')
        fig.colorbar(im, ax=axes.ravel().tolist())
        now = datetime.datetime.now()
        now_ = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(str(checkpoint_path.parent / 'bin' / f'pred_{now_}.png'))

    print('\n\n*** DONE!\n')


if __name__ == '__main__':
    main()
