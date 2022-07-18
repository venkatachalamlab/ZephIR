"""
To ease the pain of ensuring compatibility with new data structures or datasets,
this file collects key IO functions for data, metadata, and annotations
that may be edited by a user to fit their particular use case.
"""

import h5py
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path


# default getters
def get_slice(dataset: Path, t: int) -> np.ndarray:
    """Return a slice at specified index t.
    This should return a 4-D numpy array containing multi-channel volumetric data
    with the dimensions ordered as (C, Z, Y, X).
    """
    h5_filename = dataset / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return f["data"][t]


def get_annotation_df(dataset: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    with h5py.File(dataset / 'annotations.h5', 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data


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
