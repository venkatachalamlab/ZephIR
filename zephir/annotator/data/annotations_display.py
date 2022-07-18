from typing import Tuple

import cv2
import numpy as np

from .annotations_io import AnnotationTable, WorldlineTable
from .transform import mip_threeview


def hex_to_rgb(value: bytes) -> Tuple[int, ...]:
    value = value.lstrip(b'#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def draw_on_threeview(A: AnnotationTable,
                      W: WorldlineTable,
                      V: np.ndarray,
                      dot_size=2,
                      scale=(4, 1, 1)):

    threeview = mip_threeview(V, scale=scale)

    size_X = round(scale[2] * V.shape[2])
    size_Y = round(scale[1] * V.shape[1])
    size_Z = round(scale[0] * V.shape[0])

    for a in A:

        x = (a.x * size_X).astype(int)
        y = (a.y * size_Y).astype(int)
        z = (a.z * size_Z).astype(int)
        color = hex_to_rgb(W.get(a.worldline_id).color)

        # Draw on z-projection
        cv2.circle(threeview, (x, y), dot_size, color, -1)

        # Draw on x-projection
        cv2.circle(threeview, (size_X + z, y), dot_size, color, -1)

        # Draw on y-projection
        cv2.circle(threeview, (x, size_Y + z), dot_size, color, -1)

    return threeview