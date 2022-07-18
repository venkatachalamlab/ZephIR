from logging import Logger
from pathlib import Path
from typing import List
import re

import numpy as np  # needed in eval

from ..data import (AnnotationTable, WorldlineTable)
from ._utilities import default_args

@default_args("_this, _now, *, _ACTIVE")
def change_provenance(dataset: Path, annotations: AnnotationTable,
                      worldlines: WorldlineTable, window_state, arg: str,
                      logger: Logger) -> List[dict]:
    """
    arg: worldline_id, times, old_provenance, new_provenance
        worldline_id: worldline_id to delete.
            '*': all worldlines
            '_this': currently active worldline [default]
        times: times to delete.
            '*': all times
            '3:5': times 3 and 4
            '3:_now': times 3 through current time (minus 1)
        old_provenance: Any regexp for matching.
            _ACTIVE: Use the currently active provenance [default]
            '*': all provenances
            'XX..' Match XX followed by any two characters.
        new_provenance: Any regexp for matching.
            _ACTIVE: Use the currently active provenance [default]
            'XX..' Match XX followed by any two characters.
    ex: "*, _now, NEIR, ANTT": Updates all neurons in this frame with provenance "NEIR" to "ANTT"
    """

    arg_list = arg.replace(" ", "").split(",")
    bare_worldline = arg_list[0]
    bare_times = arg_list[1].replace("_now", str(window_state["t_idx"]))
    bare_old_provenance = arg_list[2].replace("_ACTIVE",
                                              window_state["active_provenance"])
    bare_new_provenance = arg_list[3].replace("_ACTIVE",
                                              window_state["active_provenance"])

    if bare_worldline == "*":
        bad_wl = True
    elif bare_worldline == "_this":
        bad_wl = annotations.df["worldline_id"] == window_state[
            "selected_worldline"]
    else:
        bad_wl = annotations.df["worldline_id"] == int(bare_worldline)

    if bare_times == "*":
        bad_times = True
    else:
        all_times = range(window_state["shape_t"])
        slice_ = eval(f"np.s_[{bare_times}]")
        times = all_times[slice_]
        if type(times) is int:
            bad_times = annotations.df["t_idx"] == times
        else:
            bad_times = annotations.df["t_idx"].apply(lambda x: x in times)

    if bare_old_provenance == "*":
        bad_prov = True
    else:
        pattern = re.compile(bare_old_provenance)
        bad_prov = annotations.df["provenance"].apply(
            lambda x: bool(pattern.match(x.decode())))

    to_change = annotations.df[bad_wl & bad_times & bad_prov].index

    annotations.df.provenance[to_change] = bare_new_provenance.encode('utf-8')

    return [{"type": "annotations/get_annotations"}]