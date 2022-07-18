from logging import Logger
from pathlib import Path
from typing import List

from ..data import AnnotationTable, WorldlineTable
from ._utilities import default_args


@default_args("0")
def jump_to_frame(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    Jump to the specified frame (starting with 0).

    Args: frame_number

        frame_number: [default 0]
    """

    arg_list = arg.replace(" ", "").split(",")
    t = int(arg_list[0])

    return [
        {
            "type": "annotation_window/set_t_idx",
            "payload": t
        },
    ]
