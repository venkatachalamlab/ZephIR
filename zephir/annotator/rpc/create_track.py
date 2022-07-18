from logging import Logger
from pathlib import Path
from typing import List

from ..data import (AnnotationTable, WorldlineTable,
                            Worldline, get_nearby_max)

from ._utilities import default_args


def create_track(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    Create a new track and select it as the current track.

    """

    w = worldlines.insert(Worldline())

    return [
        {
            "type": "worldlines/get_worldlines"
        },
        {
            "type": "annotation_window/set_selected_worldline_local",
            "payload": str(w.id)
        },
    ]
