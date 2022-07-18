from logging import Logger
from pathlib import Path
from typing import List

from ..data import AnnotationTable, WorldlineTable, Annotation

def insert_annotation(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    Insert an annotation at the current coordinates using the current track.
    """

    logger.info("arg: {}".format(arg))

    A = Annotation(
        t_idx=window_state["t_idx"],
        x=window_state["x"],
        y=window_state["y"],
        z=window_state["z"],
        worldline_id=window_state["selected_worldline"] or 0,
        parent_id=0,
        provenance=window_state["active_provenance"]
    )
    a = annotations.insert(A)

    return [{
            "type": "annotations/get_annotations"
        },
        {
            "type": "annotation_window/set_selected_annotation_local",
            "payload": int(a.id)
        }]
