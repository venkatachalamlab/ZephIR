"""
This file collects key functions for preprocessing data and annotations.
"""

import datetime
from functools import lru_cache, partial
import h5py
import importlib.util
import json
import numpy as np
import pathlib
from pathlib import Path
import platform
import torch
from tqdm import tqdm
from typing import Optional, Callable
from types import ModuleType


# from .getters import *
from . import getters as default_getters
from ..__version__ import __version__


def _module_name_from_path(p: Path) -> str:
    """Return a globally unique module name associated with a given
    directory."""
    return str(p / "getters")


def _getter_module_from_path(dataset: Path) -> Optional[ModuleType]:
    module_name = _module_name_from_path(dataset)
    getter_file = dataset / "getters.py"

    if not getter_file.is_file():
        return None

    spec = importlib.util.spec_from_file_location(module_name, getter_file)

    getters = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(getters)  # type: ignore

    return getters


def _make_getter_with_default(name: str, default: Callable) -> Callable:
    """This hideous function creates a getter with the given name. If that
    function doesn't exist in a "getter.py" file within the dataset, then
    a specified default function is called. The default is bound with
    the path."""

    def _getter(p: Path, *getter_args):
        getters = _getter_module_from_path(p)

        if getters is None or not hasattr(getters, name):
            _fn = partial(default, p)

        else:
            _fn = partial(getters.__getattribute__(name), p)

        return _fn(*getter_args)

    _getter.__name__ = name

    return _getter


get_slice = _make_getter_with_default('get_slice', default_getters.get_slice)
get_metadata = _make_getter_with_default('get_metadata', default_getters.get_metadata)
get_annotation_df = _make_getter_with_default('get_annotation_df', default_getters.get_annotation_df)


@lru_cache()
def get_data(dataset, t, g=1, c=None):
    data = get_slice(dataset, t).astype(float)
    data = np.power(data / np.max(data), 1/g)
    if len(data.shape) == 3:
        return data[np.newaxis, ...]
    elif c is None:
        return data
    elif c >= 0:
        return data[c, np.newaxis, ...]
    elif c == -1:
        return np.max(data, axis=0)[np.newaxis, ...]
    return data


def get_annotation(annotation, t, exclusive_prov=None, exclude_self=True):
    annot = np.stack(
        [(annotation['x'][annotation['t_idx'] == t]).astype(float) * 2 - 1,
         (annotation['y'][annotation['t_idx'] == t]).astype(float) * 2 - 1,
         (annotation['z'][annotation['t_idx'] == t]).astype(float) * 2 - 1
         ], axis=-1
    )
    worldline_id = annotation['worldline_id'][annotation['t_idx'] == t]
    provenance = np.array(annotation['provenance'][annotation['t_idx'] == t])
    if exclusive_prov is not None:
        annot = annot[provenance == exclusive_prov, :]
        worldline_id = worldline_id[provenance == exclusive_prov]
        provenance = provenance[provenance == exclusive_prov]
    elif exclude_self:
        annot = annot[provenance != b'ZEIR', :]
        worldline_id = worldline_id[provenance != b'ZEIR']
        provenance = provenance[provenance != b'ZEIR']

    u, i, c = np.unique(worldline_id, return_index=True, return_counts=True)
    ovc_idx = np.where(c > 1)[0]
    for j in ovc_idx:
        i[j] = np.where(worldline_id == u[j])[0][-1]
    return u, annot[i, ...], provenance[i]


def get_checkpoint(path, key: str, verbose=False):
    """
    Retrieves given key and value from an existing checkpoint.
    """

    checkpoint = load_checkpoint(path, verbose=verbose)

    if key in checkpoint.keys():
        value = checkpoint[key]
        del checkpoint
        return value
    elif verbose:
        print(f'*** ERROR: {key} not found in checkpoint!')
    del checkpoint
    return None


def update_checkpoint(path, dictionary: dict, verbose=True):
    """
    Updates an existing checkpoint dictionary with given key and value and saves to file.
    """

    checkpoint = load_checkpoint(path, verbose=False)

    for key in dictionary.keys():
        checkpoint[key] = dictionary[key]

        if key == 'args':
            with open(Path(path) / 'args.json', 'w') as outfile:
                json.dump(dictionary[key], outfile, indent=4)

    checkpoint['last_update'] = str(datetime.datetime.now())

    with open(Path(path) / 'checkpoint.pt', 'wb') as f:
        torch.save(checkpoint, f)

    if verbose:
        print(f'Checkpoint updated for {list(dictionary.keys())} '
              f'@ [{checkpoint["last_update"]}]')

    del checkpoint
    return


def load_checkpoint(path, fallback=True, verbose=True):
    if verbose:
        print('Loading checkpoint...')

    file_name = Path(path) / 'checkpoint.pt'
    checkpoint, map_loc = None, None
    if file_name.is_file():

        while checkpoint is None:

            try:
                checkpoint = torch.load(str(file_name), map_location=map_loc)

            except ModuleNotFoundError:
                print('*** ERROR: checkpoint not compatible with current ZephIR version!')
                if fallback:
                    print('Creating empty checkpoint...')
                    checkpoint = {'__version__': __version__}
                else:
                    exit()

            except (NotImplementedError, RuntimeError) as e:
                if platform.system() == 'Darwin' or platform.system() == 'Linux':
                    pathlib.WindowsPath = pathlib.PosixPath
                elif platform.system() == 'Windows':
                    pathlib.PosixPath = pathlib.WindowsPath

            except RuntimeError:
                print('*** CUDA NOT AVAILABLE! Mapping CUDA tensors to CPU...')
                map_loc = torch.device('cpu')

    elif fallback:
        print('*** CHECKPOINT NOT FOUND! Creating empty checkpoint...')
        return {'__version__': __version__}

    else:
        print('*** CHECKPOINT NOT FOUND!')
        exit()

    if verbose and '__version__' not in checkpoint.keys():
        print('*** WARNING: __version__ not found in checkpoint! '
              'Setting to current ZephIR version, but may not be accurate. '
              'Consider resetting checkpoint.\n')
        checkpoint['__version__'] = __version__
    elif verbose and __version__ != checkpoint['__version__']:
        print('*** WARNING: checkpoint does not match current ZephIR version! '
              'Consider resetting checkpoint.\n')
    elif verbose:
        print('Successfully loaded checkpoint!\n')

    return checkpoint


def update_metadata(path, dictionary: dict):
    """Update an existing metadata file with given key and value."""
    file_name = Path(path) / 'metadata.json'

    try:
        metadata = get_metadata(path)
    except (OSError, IOError) as e:
        print('*** WARNING: Error loading metadata!')
        if not file_name.is_file():
            print(f'Creating empty metadata\t'
                  f'-- properly populate this file with the right metadata!\n'
                  f'Saving to {file_name}...\n')
            metadata = {}
        else:
            exit()

    for key in dictionary.keys():
        metadata[key] = dictionary[key]

    with open(file_name, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)
