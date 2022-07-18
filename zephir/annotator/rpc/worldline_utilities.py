from logging import Logger
from pathlib import Path
from typing import List

from ..data import (AnnotationTable, WorldlineTable,
                    cleanup_worldlines, color_worldlines)


def renumber_worldlines(dataset: Path, annotations: AnnotationTable,
                        worldlines: WorldlineTable, window_state, arg: str,
                        logger: Logger) -> List[dict]:
    """
    Remove unused worldlines and renumber them consecutively starting from 1.
    """

    (annotations, worldlines) = cleanup_worldlines(annotations, worldlines)

    return [{
        "type": "annotations/get_annotations"
    }, {
        "type": "worldlines/get_worldlines"
    }]


def randomly_color_worldlines(dataset: Path, annotations: AnnotationTable,
                              worldlines: WorldlineTable, window_state,
                              arg: str, logger: Logger) -> List[dict]:
    """
    Randomly color all worldlines.
    """

    color_worldlines(worldlines)

    return [
        {
            "type": "worldlines/get_worldlines"
        },
    ]
