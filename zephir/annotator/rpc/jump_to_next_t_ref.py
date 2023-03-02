from logging import Logger
from pathlib import Path
from typing import List
import json

from ...utils.io import get_metadata
from ..data import AnnotationTable, WorldlineTable
from ._utilities import default_args


@default_args("False")
def jump_to_next_t_ref(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    Jumps to the next frame in the list of reference frames stored in the metadata.

    Args: reverse

        reverse: Specifies the direction of the jump.

            'False': To jump between frames in ascending order [default]
            'True': To jump between frames in descending order
    """

    arg_list = arg.replace(" ", "").split(",")
    step = -1 if arg_list[0] == "True" else 1

    metadata = get_metadata(dataset)
    t_list = metadata['t_ref']
    t_list.sort()
    t_now = window_state["t_idx"]

    if t_now in t_list:
        t = t_list[(t_list.index(t_now) + step) % len(t_list)]
    else:
        diff = lambda t_list : abs(t_list - t_now)
        t = min(t_list, key=diff)

    return [
        {
            "type": "annotation_window/set_t_idx",
            "payload": t
        },
    ]
