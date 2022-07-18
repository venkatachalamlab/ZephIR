from logging import Logger
from pathlib import Path
from typing import List

from ..data import (get_channel_specific_slice_3D, Annotation, Worldline,
                    AnnotationTable, WorldlineTable, get_nearby_max)
from ._utilities import default_args


@default_args("0.01, 0.08, 0.005, 0.04, False, _ACTIVE, *")
def insert_local_max(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    Insert an annotation at the local max near the current point.

    Args: r_x, r_z, blur_x, blur_z, new_worldline, provenance, channel

        r_x: Search radius in x, normalized units (0, 1). The radius in y is
            set by the aspect ratio.
        r_z: Search radius in z, normalized units (0, 1). It should be larger
            than r_x because of the typically smaller dimension.
        blur_x: The sigma of a gaussian blur to apply prior to finding the
            local maximum. blur_y is calculated using the aspect ratio.
        blur_z: Same for z.
        new_worldline: Create a new worldline/track for this annotation.
        provenance: Anything is okay. Special values:
            _ACTIVE: Use the currently active provenance [default]
        channel: channel to use to calculate local max:
            *: All channles [default]
    """

    aspect_ratio = window_state["shape_x"]/window_state["shape_y"]

    arg_list = arg.replace(" ", "").split(",")
    r_x = float(arg_list[0])
    r_y = r_x * aspect_ratio
    r_z = float(arg_list[1])
    blur_x = float(arg_list[2])
    blur_y = blur_x * aspect_ratio
    blur_z = float(arg_list[3])
    new_worldline = arg_list[4] == "True"
    provenance = arg_list[5]
    channel = arg_list[6]

    if provenance == "_ACTIVE":
        provenance = window_state["active_provenance"]

    vol = get_channel_specific_slice_3D(dataset=dataset, t=window_state["t_idx"], channel=channel)
    center = (window_state["z"], window_state["y"], window_state["x"])
    radius = (r_z, r_y, r_x)
    blur_sigma = (blur_z, blur_y, blur_x)

    new_coords = get_nearby_max(vol, center, radius, blur_sigma) # type: ignore

    if new_worldline:
        worldline = worldlines.insert(Worldline())
        worldline_id = worldline.id

    else:
        worldline_id = window_state["selected_worldline"] or 0

    A = Annotation(
        t_idx=window_state["t_idx"],
        x=new_coords[2],
        y=new_coords[1],
        z=new_coords[0],
        worldline_id=worldline_id,
        parent_id=0,
        provenance=provenance
    )
    a = annotations.insert(A)

    return [
        {"type": "annotations/get_annotations"},
        {"type": "worldlines/get_worldlines"},
        {
            "type": "annotation_window/set_selected_annotation_local",
            "payload": int(a.id)
        }
    ]
