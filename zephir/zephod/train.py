"""
Training ZephOD feature detection network

Usage:
    train_zephod -h | --help
    train_zephod -v | --version
    train_zephod --index=<index> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --index=<index>                     path to json index of data directories to use as training data.
    --batch_size=<batch_size>           batch size in volumes. [default: 1]
    --channel=<channel>                 data channel to use as input.
    --cuda=<cuda>                       check if a CUDA-compatible GPU is available for  use. [default: True]
    --loss_func=<loss_func>             loss function to use. [default: BCE]
    --lr_init=<lr_init>                 learning rate for optimizer. [default: 1.0]
    --model=<model>                     name of model being trained.
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
        index=Path('.'),
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

    with open(index) as f:
        _index = json.load(f)
        dataset_paths = [Path(p) for p in _index.values()]

    annots_agg = []
    t_note_agg = []
    isolates_agg = []
    for dataset in dataset_paths:
        annotation = get_annotation_df(dataset)
        t_note = np.unique(annotation['t_idx']).astype(int)
        t_note = np.sort(t_note)
        nn_list = [len(get_annotation(annotation, t)[0]) for t in t_note]
        n_neuron, t_neuron = np.max(nn_list), t_note[np.argmax(nn_list)]
        worldline_id, _, _ = get_annotation(annotation, t_neuron)
        print(f'\nUsing frame #{t_neuron} as initial reference with {n_neuron} annotations found...')

        metadata = get_metadata(dataset)
        shape_t = metadata['shape_t']
        xyz_note = []
        for t in t_note:
            u, annot, prov = get_annotation(annotation, t)
            u_idx = np.array([np.where(u == w)[0][-1] for w in worldline_id if w in u], dtype=int)
            annot = annot[u_idx, ...]
            if (t >= shape_t
                    or annot.shape[0] > n_neuron
                    or annot.shape[0] < n_neuron
                    or annot.shape[0] == 0
                ):
                t_note = np.setdiff1d(t_note, [t])
                continue
            xyz_note.append(annot)
        print(f'Annotations loaded for frames {t_note} with shape: {np.array(xyz_note).shape}')

        annots = np.array(xyz_note)
        annots_agg.append(annots)
        t_note_agg.append(t_note)

        pbar = tqdm(
            range(len(t_note)),
            desc='Finding isolated features',
            unit='frames',
        )
        for i in pbar:
            # for i in range(len(t_note)):
            data = get_data(dataset, t_note[i], c=channel)
            isolated_neurons = identify(data, annots[i], 12)
            for n in isolated_neurons:
                isolates_agg.append(n)
            pbar.set_postfix(Neurons=f'{len(isolates_agg)}')

    print(f'\n\nAnnotations loaded for a total of {len(annots_agg)} datasets!'
          f'\n*** Found a total of {len(isolates_agg)} isolated neurons')

    print('\nCompiling model...')
    vol = get_data(dataset_paths[0], t_note_agg[0][0], c=channel)
    if model_kwargs is None:
        model_kwargs = {
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
    scaler = torch.cuda.amp.GradScaler()
    if loss_func == 'BCE':
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = corr_loss

    n_val = max(min(n_val, len(t_note_agg) - 1), 0)
    t_list = t_note_agg[:-n_val]
    t_val = t_note_agg[-n_val:]

    print('\n\n\n******* BEGIN TRAINING *******\n')
    model.train()
    pbar = tqdm(range(n_epoch), desc='Training model', unit='epochs')
    for epoch in pbar:
        # for epoch in range(n_epoch):
        tpbar = tqdm(range(n_trn), leave=False, desc='Training samples', unit='vol')
        for _ in tpbar:
            # for i in range(n_trn):
            with torch.no_grad():
                d_idx = random.randint(0, len(t_list) - 1)
                dataset = dataset_paths[d_idx]
                vol = get_data(dataset, t_note_agg[d_idx][0], c=channel)

                input_list, target_list = [], []
                for j in range(batch_size):
                    t_idx = random.randint(0, len(t_note_agg[d_idx]) - 1)
                    synth, labels = generate_synthetic_data(
                        vol, annots_agg[d_idx][t_idx], isolates_agg, True, True
                    )
                    input_list.append(to_tensor(synth, dev=dev))
                    target_list.append(to_tensor(labels, dev=dev))
                input_tensor, target_tensor = torch.stack(input_list, dim=0).to(torch.float16), torch.stack(target_list, dim=0).to(torch.float16)

            optimizer.zero_grad()
            with torch.autocast(device_type=dev):
                pred = model(input_tensor)
                loss = loss_function(pred, target_tensor)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                tpbar.set_postfix(Loss=f'{loss.item():.4f}')
                pbar.set_postfix(Loss=f'{loss.item():.4f}')
    
    with torch.no_grad():
        checkpoint = {
            'model_kwargs': model_kwargs,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
        }

        input_val = target_val = val_pred = None
        if n_val > 0:
            model.eval()
            input_val, target_val = [], []
            for i in tqdm(range(len(t_val)), leave=False, desc='Validating samples', unit='vol'):
                # for i in range(len(t_val)):
                dataset = dataset_paths[-n_val + i]
                vol = get_data(dataset, t_note_agg[-n_val + i][0], c=channel)
                for j in range(len(t_note_agg[-n_val + i])):
                    synth, labels = generate_synthetic_data(
                        vol, annots_agg[-n_val + i][j], isolates_agg, True, True
                    )
                    input_val.append(to_tensor(synth, dev=dev))
                    target_val.append(to_tensor(labels, dev=dev))
            input_val, target_val = torch.stack(input_val, dim=0), torch.stack(target_val, dim=0)
            val_pred = model(input_val)
            val_loss = loss_function(val_pred, target_val)
            print(f'\nValidation loss: {val_loss.item():.4f}\n\n')

    return checkpoint, (input_val, target_val, val_pred), (to_numpy(model.alpha), to_numpy(model.gamma))


def main():
    args = docopt(__doc__, version=f'ZephOD {__version__}')
    print(args, '\n')

    if torch.cuda.is_available() and args['--cuda'] in ['True', 'Y', 'y']:
        # Moving to GPU
        print('\n*** GPU available!\n')
        dev = 'cuda'
    else:
        dev = 'cpu'

    if args['--model'] is None:
        name = 'model'
    else:
        name = args['--model']
    checkpoint_path = Path(__file__).parent / f'{name}.pt'

    override = args['--override'] in ['True', 'Y', 'y']
    checkpoint = {
        'model_kwargs': None,
        'state_dict': None,
        'opt_dict': None,
        'loss_list': None
    }
    if Path.is_file(checkpoint_path):
        print('\nPrevious checkpoint available.')
        if not (checkpoint_path.parent / 'backup').is_dir():
            Path.mkdir(checkpoint_path.parent / 'backup')
        now = datetime.datetime.now()
        _now = now.strftime("%m_%d_%Y_%H_%M")
        shutil.copy(
            checkpoint_path,
            checkpoint_path.parent / 'backup' / f'{name}_{_now}.pt'
        )
        if not override:
            try:
                checkpoint = torch.load(checkpoint_path)
            except RuntimeError:
                print('*** CUDA NOT AVAILABLE! Mapping CUDA tensors to CPU...')
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    checkpoint, validation, params = train_model(
        index=Path(args['--index']),
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

    input_val, target_val, val_pred = validation
    if val_pred is not None:
        print('\nSaving prediction to a new file...')
        if not (checkpoint_path.parent / 'bin').is_dir():
            Path.mkdir(checkpoint_path.parent / 'bin')
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        axes[0].imshow(np.max(np.max(to_numpy(input_val[0]), axis=0), axis=0), vmin=0, vmax=1)
        axes[0].set_title('Input Volume')
        axes[1].imshow(np.max(to_numpy(val_pred[0][0]), axis=0), vmin=0, vmax=1)
        axes[1].set_title('Output Labels')
        im = axes[2].imshow(np.max(to_numpy(target_val[0][0]), axis=0), vmin=0, vmax=1)
        axes[2].set_title('GT Labels')
        fig.colorbar(im, ax=axes.ravel().tolist())
        now = datetime.datetime.now()
        _now = now.strftime("%m_%d_%Y_%H_%M")
        plt.savefig(str(checkpoint_path.parent / 'bin' / f'{name}_pred_{_now}.png'))

    print('\n\n*** DONE!\n')


if __name__ == '__main__':
    main()
