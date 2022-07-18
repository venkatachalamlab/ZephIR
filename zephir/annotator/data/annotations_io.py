# An annotation is a coordinate in a 4D array. Worldlines are lists of
# annotations.
#
# author: vivekv2@gmail.com

import base64
import colorsys
from dataclasses import dataclass, asdict, field
from functools import lru_cache, reduce
import json
from pathlib import Path
import uuid
from typing import Tuple, Optional

import h5py
import numpy as np
import pandas as pd

from .dataclass_helpers import DataclassTableBase
from ...utils.getters import get_annotation_df


def base64str_from_uint32(x: np.uint32):
    return base64.standard_b64encode(x.tobytes())[:6]


def get_random_base64_id():
    idu32 = np.uint32(uuid.uuid1().time_low)
    return base64str_from_uint32(idu32)


_S4 = np.dtype("S4")
_S7 = np.dtype("S7")


@dataclass
class Annotation:
    id: np.uint32 = 0  # 0 is unassigned. Get an id after inserting into table.
    t_idx: np.uint32 = 0  # in [0, shape_t)
    x: np.float32 = 0.5  # in (0, 1)
    y: np.float32 = 0.5  # in (0, 1)
    z: np.float32 = 0.5  # in (0, 1)
    worldline_id: np.uint32 = 0
    parent_id: np.uint32 = 0  # =0 is null/none
    provenance: _S4 = b"NULL"  # unknown

    def __post_init__(self):

        self.id = np.uint32(self.id)
        self.t_idx = np.uint32(self.t_idx)
        self.x = np.float32(self.x)
        self.y = np.float32(self.y)
        self.z = np.float32(self.z)
        self.worldline_id = np.uint32(self.worldline_id)
        self.parent_id = np.uint32(self.parent_id)
        self.provenance = np.string_(self.provenance)

        # if self.id == b"":
        #     self.id = get_random_base64_id()

    def to_tuple(self) -> tuple:
        return (self.id, self.t_idx, self.x, self.y, self.z, self.worldline_id,
                self.parent_id, self.provenance)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonable_dict(self) -> dict:

        jsonable_dict = {}
        for k, v in self.to_dict().items():
            if type(v) is bytes or type(v) is np.string_:
                v = v.decode()
            if type(v) is np.uint32:
                v = int(v)
            if type(v) is np.float32:
                v = str(v)
            jsonable_dict[k] = v

        return jsonable_dict


class AnnotationTable(DataclassTableBase):
    row_class = Annotation

    def get_t(self, t_idx: int):
        return self.filter(lambda x: x["t_idx"] == t_idx)

    def to_jsonable_dict(self) -> dict:

        jsonable_dict = {}
        for annotation in self:
            jsonable_dict[str(annotation.id)] = annotation.to_jsonable_dict()

        return jsonable_dict

    @staticmethod
    def from_hdf(filename: Path):

        # filename = Path(filename)
        #
        # f = h5py.File(filename, "r")
        # data = pd.DataFrame()
        #
        # # Handle 'correctly' shaped data from Pythons H5 library.
        # if len(f["id"].shape) == 1:
        #
        #     for k in f:
        #         data[k] = f[k]
        #
        #     return AnnotationTable(data)
        #
        # # Matlab cannot make 1D arrays, and it cannot save char arrays.
        # # Handle both of these here
        # else:
        #     for k in f:
        #         vals = np.squeeze(f[k])
        #         if k == "provenance":
        #             vals = np.array(
        #                 list(map(lambda x: x.tobytes(), vals)))
        #         data[k] = vals
        #
        #     return AnnotationTable(data)

        return AnnotationTable(get_annotation_df(Path(filename)))


@dataclass
class Worldline:
    id: np.uint32 = 0
    name: np.string_ = b"null"
    color: _S7 = b"#ffffff"

    def __post_init__(self):

        self.name = np.string_(self.name)
        self.color = np.string_(self.color)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonable_dict(self):

        jsonable_dict = {}
        for k, v in self.to_dict().items():
            if type(v) is bytes or type(v) is np.string_:
                v = v.decode()
            if type(v) is np.uint32:
                v = int(v)
            if type(v) is np.float32:
                v = str(v)
            jsonable_dict[k] = v

        return jsonable_dict


class WorldlineTable(DataclassTableBase):
    row_class = Worldline

    def to_jsonable_dict(self) -> dict:

        jsonable_dict = {}
        for worldline in self:
            jsonable_dict[str(worldline.id)] = worldline.to_jsonable_dict()

        return jsonable_dict

    @staticmethod
    def from_annotations(annotations: AnnotationTable):

        worldlines = WorldlineTable()
        worldlines._insert_and_preserve_id(Worldline(id=0))

        worldline_ids = np.unique(annotations.df["worldline_id"])
        for id in worldline_ids:
            worldlines.insert(Worldline(name=str(id)))

        return worldlines


@lru_cache()
def get_all_neuron_data() -> dict:

    file_path = Path(__file__).parent.absolute()
    neuron_list_file = file_path / "./neurons_celegans.json"
    print(file_path)
    with open(neuron_list_file) as fp:
        data = json.load(fp)
    return data


def get_neuron_data(x: str) -> dict:
    return get_all_neuron_data()["neurons"][x]


def cleanup_worldlines(A: AnnotationTable, W: WorldlineTable
                       ) -> Tuple[AnnotationTable, WorldlineTable]:
    """Sequentially renumber all annotations and worldlines, and remake the
    worldline table."""

    new_ids = range(1, len(A) + 1)
    old_ids = A.df.id
    update_aid = {old_ids[i]: new_ids[i] for i in range(len(new_ids))}
    update_aid[0] = 0

    A.df.id = A.df.id.apply(lambda x: update_aid[x])
    A.df.parent_id = A.df.parent_id.apply(lambda x: update_aid[x])

    used_W = np.unique(A.df.worldline_id)
    N = len(used_W)
    new_ids = range(1, N + 1)
    update_wid = {used_W[i]: new_ids[i] for i in range(N)}
    update_wid[0] = 0

    A.df.worldline_id = A.df.worldline_id.apply(lambda x: update_wid)

    W_new = WorldlineTable()
    W_new._insert_and_preserve_id(Worldline())

    for i in range(N):

        if used_W[i] in W.df.id:
            wline = W.get(used_W[i])
            wline.id = new_ids[i]
        else:
            wline = Worldline(id=new_ids[i])

        W_new._insert_and_preserve_id(wline)

    return (A, W_new)


def color_worldlines(W: WorldlineTable) -> None:
    """Overwrite the color of all worldlines using evenly spaced hues and
    random lightness / saturation, both high."""
    def _get_N_colors(N):
        colors = []
        for i in range(N):
            h = ((i * 157) % 360) / 360.
            l = (60 + 10 * np.random.rand()) / 100.
            s = (90 + 10 * np.random.rand()) / 100.
            rgb = colorsys.hls_to_rgb(h, l, s)
            rgb_256 = map(lambda x: max(0, min(int(x * 255), 255)), rgb)
            rgb_code = "#{0:02x}{1:02x}{2:02x}".format(*rgb_256)
            colors.append(rgb_code)
        return colors

    colors = _get_N_colors(len(W))
    W.df.color = colors


def load_annotations(dataset: Optional[Path] = None
                     ) -> Tuple[AnnotationTable, WorldlineTable]:

    if dataset is None:
        dataset = Path(".")

    annotation_file = dataset / "annotations.h5"

    if annotation_file.exists():
        annotations = AnnotationTable.from_hdf(dataset)
    else:
        annotations = AnnotationTable()

    worldline_file = dataset / "worldlines.h5"
    if worldline_file.exists():
        worldlines = WorldlineTable.from_hdf(worldline_file)
    else:
        worldlines = WorldlineTable.from_annotations(annotations)

    return (annotations, worldlines)


def save_annotations(annotations: AnnotationTable,
                     worldlines: WorldlineTable,
                     dataset: Path = None) -> None:

    if dataset is None:
        dataset = Path(".")

    annotations.to_hdf(dataset / "annotations.h5")
    worldlines.to_hdf(dataset / "worldlines.h5")


def stash_annotations(annotations: AnnotationTable,
                      worldlines: WorldlineTable,
                      dataset: Path = None) -> None:

    if dataset is None:
        dataset = Path(".")

    annotations.to_hdf(dataset / "annotations_unsaved.h5")
    worldlines.to_hdf(dataset / "worldlines_unsaved.h5")
