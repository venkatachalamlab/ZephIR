# These are tools to be used along with Python dataclasses.
#
# Author: vivekv2@gmail.com

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Type, TypeVar

import pandas as pd
import numpy as np
import h5py


def add_backup_timestamp(path: Path) -> Path:
    timestamp = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    backup_folder = path.parent / "backup"
    backup_folder.mkdir(exist_ok=True)
    return backup_folder / (path.stem + timestamp + path.suffix)


def archive_file(path: Path):
    if path.exists():
        path.rename(add_backup_timestamp(path))


DT = TypeVar('DT', bound='DataclassTableBase')  # For classmethod typing.


class DataclassTableBase:
    """This class needs to be subclassed, and a valid dataclass has to be
    assigned to row_class."""

    row_class: type

    def __init__(self, data: pd.DataFrame = None):

        self.column_types = dict(self.row_class.__annotations__)

        self.df = pd.DataFrame(columns=self.column_types)

        if data is not None:
            self.df = pd.concat([self.df, data])

        self.next_id = np.max(self.df['id']) + 1
        if np.isnan(self.next_id):
            self.next_id = 1

        self._fix_types()

    def _fix_types(self):
        """The index type may be coerced by pandas (e.g. u64 instead of u32),
        so that needs to be set after resetting it."""

        for k in self.df.columns:
            self.df[k] = self.df[k].astype(self.column_types[k])

    def _insert_and_preserve_id(self, data):

        row = pd.DataFrame(asdict(data), index=(0, ))
        self.df = pd.concat([self.df, row], ignore_index=True)
        return data

    def insert(self, data):

        data.id = np.uint32(self.next_id)
        row = pd.DataFrame(asdict(data), index=(0, ))
        self.df = pd.concat([self.df, row], ignore_index=True)
        self.next_id = self.next_id + 1
        return data

    def object_from_row(self, row: pd.DataFrame):
        row_dict = {k: list(v.items())[0][1] for k, v in row.to_dict().items()}
        return self.row_class(**row_dict)

    def get_row(self, idx: int):

        row = self.df.loc[(idx, ), :]
        return self.object_from_row(row)

    def get_first(self):

        row = self.df.iloc[[0,], :]
        return self.object_from_row(row)

    def get_row_idx(self, id: int):

        row = self.filter(lambda x: x["id"] == id).df
        return row.index[0]

    def get(self, id: int):

        row = self.filter(lambda x: x["id"] == id)
        return self.object_from_row(row.df)

    def update(self, id: int, data: dict):

        row_idx = self.get_row_idx(id)

        new_data = asdict(self.get(id))
        new_data.update(data)
        new_row = pd.DataFrame(new_data, index=(row_idx, ))
        self.df.update(new_row)

        return self.get(id)

    def delete(self, id: int):

        row_idx = self.get_row_idx(id)
        self.df = self.df.drop(row_idx)

        return id

    def delete_ids(self, ids: list):

        rows_to_delete = self.df[self.df["id"].apply(lambda x: x in ids)].index
        self.df = self.df.drop(rows_to_delete)

    def filter(self: DT, fn) -> DT:

        df = self.df[fn(self.df)]
        return type(self)(df)

    def to_hdf(self, filename: Path, archive=True):

        filename = Path(filename)

        if archive:
            archive_file(filename)

        f = h5py.File(filename, "w")
        N = self.df.shape[0]

        for c in self.df.columns:

            if self.column_types[c] == np.string_:
                dtype = "S8"
            else:
                dtype = self.column_types[c]

            f.create_dataset(c,
                             shape=(N, ),
                             dtype=dtype,
                             data=self.df[c].astype(dtype))

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return repr(self.df)

    def _repr_html_(self):
        return self.df._repr_html_()

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < len(self.df):
            val = self.get_row(self.df.index[self._iter_idx])
            self._iter_idx = self._iter_idx + 1
            return val
        else:
            raise StopIteration

    @classmethod
    def from_hdf(cls: Type[DT], filename: Path) -> DT:

        filename = Path(filename)

        f = h5py.File(filename, "r")

        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]

        return cls(data)
