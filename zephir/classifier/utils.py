import h5py
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path


def get_slice(path: Path, t: int) -> np.ndarray:
    """Return a slice at specified index t.
    This should return a 4-D numpy array containing multi-channel volumetric data
    with the dimensions ordered as (C, Z, Y, X).
    """
    f = h5py.File(path / "data.h5", 'r')
    return f["data"][t]


def get_worldlines_df(path: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    with h5py.File(path / 'worldlines.h5', 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data


def get_annotation_df(path: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    with h5py.File(path / 'annotations.h5', 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data


def get_annotation(annotation_df, t, prov=None):
    """Give an annotation DataFrame and a selected time index(, and a provenance to filter for).
    This returns the following:
    - unique_worldline_id: sorted list of unique worldline_id's
    - annotation: annotation coordinates (-1 to 1), sorted by worldline_id (same as u)
    """
    annotation = np.stack(
        [(annotation_df['x'][annotation_df['t_idx'] == t]).astype(float) * 2 - 1,
         (annotation_df['y'][annotation_df['t_idx'] == t]).astype(float) * 2 - 1,
         (annotation_df['z'][annotation_df['t_idx'] == t]).astype(float) * 2 - 1
         ], axis=-1
    )
    worldline_id = annotation_df['worldline_id'][annotation_df['t_idx'] == t]
    provenance = np.array(annotation_df['provenance'][annotation_df['t_idx'] == t])
    if prov is not None:
        annotation = annotation[provenance == prov, :]
        worldline_id = worldline_id[provenance == prov]

    unique_worldline_id, i, c = np.unique(worldline_id, return_index=True, return_counts=True)
    ovc_idx = np.where(c > 1)[0]
    for j in ovc_idx:
        i[j] = np.where(worldline_id == unique_worldline_id[j])[0][-1]

    return unique_worldline_id, annotation[i, ...]


def get_times(path: Path) -> np.ndarray:
    """Return the timestamp of a given data frame."""
    f = h5py.File(path / "data.h5", 'r')
    return f["times"][:]


def get_metadata(dataset: Path) -> dict:
    """Load and return metadata for the dataset as a Python dictionary.
    This should contain at least the following:
    - shape_t
    - shape_c
    - shape_z
    - shape_y
    - shape_x
    """
    json_filename = dataset / "metadata.json"
    with open(json_filename) as json_file:
        metadata = json.load(json_file)
    return metadata


def update_metadata(path, dictionary: dict):
    """Update an existing metadata file with given key and value."""
    file_name = Path(path) / 'metadata.json'

    try:
        metadata = get_metadata(path)
    except (OSError, IOError) as e:
        print('*** WARNING: Error loading metadata!')
        if not file_name.is_file():
            print(f'Creating empty metadata and saving to {file_name}...\n')
            metadata = {}
        else:
            return

    for key in dictionary.keys():
        metadata[key] = dictionary[key]

    with open(file_name, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)
