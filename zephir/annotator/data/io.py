# Read and write data from disk.
#
# A dataset is a single folder. To get data from the folder, you need to create
# a file called "getters.py" containing functions that return data and
# metadata.
#
# Authors: vivekv2@gmail.com

from functools import lru_cache, partial
import importlib.util
import json
from pathlib import Path
from typing import Optional, Callable
from types import ModuleType
import h5py
import numpy as np

from ...utils import io as zeir


def _module_name_from_path(p: Path) -> str:
    """Return a globally unique module name associated with a given
    directory."""
    return str(p / "getters")


@lru_cache()
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


def _get_metadata_default(dataset_path: Path):
    # json_filename = dataset_path / "metadata.json"
    # with open(json_filename) as json_file:
    #     metadata = json.load(json_file)
    # return metadata
    return zeir.get_metadata(dataset_path)


get_metadata = _make_getter_with_default("get_metadata", _get_metadata_default)


def _get_slice_default(dataset_path: Path, t: int) -> np.ndarray:
    # """Return a slice from the "data" field of an HDF5 file.
    # Time (t) is assumed to be the last (longest-stride) dimension. This should
    # generally be cached to avoid opening and closing files too many times:
    #
    #     my_get_slice = lru_cache(get_h5_slice, maxsize=1000)
    #
    # """
    # h5_filename = dataset_path / "data.h5"
    #
    # f = h5py.File(h5_filename, 'r')
    #
    # return f["data"][t]
    return zeir.get_slice(dataset_path, t)


def save_metadata_default(x: dict, dataset_path: Path) -> None:
    """Save the given dictionary as metadata.json in the given path."""

    json_filename = dataset_path / "metadata.json"
    with open(json_filename, 'w') as json_file:
        json.dump(x, json_file)


def save_dataset_default(x: np.ndarray, dataset_path: Path):
    """Save the given array in the given folder."""

    h5_filename = dataset_path / "data.h5"
    if h5_filename.is_file():
        h5_filename.unlink()
    f = h5py.File(h5_filename, "w")

    chunk_size = (1, *x.shape[1:])
    dset = f.create_dataset("data", data=x, chunks=chunk_size,
                            compression="gzip", compression_opts=7)

    f.close()


get_slice = _make_getter_with_default("get_slice", _get_slice_default)


def get_slice_3D(dataset: Path, t: int, method=np.max) -> np.ndarray:
    """Return a 3D slice by projeting extra dimensions (usually color/channel).
    By default, this will provide a maximum intensity projection, but other
    functions with the same type signature as numpy.max will work as well."""

    A = get_slice(dataset, t)

    if np.ndim(A) == 3:
        return A

    return method(A, axis=0)


def get_channel_specific_slice_3D(dataset: Path, t: int, method=np.max, channel="*") -> np.ndarray:
    """Return a 3D slice for a specific channel.
    By default, this will provide a maximum intensity projection, but other
    functions with the same type signature as numpy.max will work as well."""

    A = get_slice(dataset, t)

    if np.ndim(A) == 3:
        return A
    else:
        if channel == "*":
            return method(A, axis=0)
        else:
            return A[int(channel)]


def _get_times_default(dataset_path: Path) -> np.ndarray:
    """Return the timestamp of a given data frame."""
    h5_filename = dataset_path / "data.h5"

    f = h5py.File(h5_filename, 'r')

    return f["times"][:]


get_times = _make_getter_with_default("get_times", _get_times_default)
