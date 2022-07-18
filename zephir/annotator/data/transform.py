# Utilities to visualize images from ND arrays.
#
# Authors: vivekv2@gmail.com, maedeh.seyedolmohadesin@gmail.com

from math import floor
from typing import Tuple, List, Union

import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage.filters import convolve

from .shapes import gaussian

def blur(vol:np.ndarray, sigma:np.ndarray) -> np.ndarray:
    """Blur the given volume using a gaussian kernal with specified width. The
    shape of vol is preserved."""

    sigma = np.array(sigma)

    g = gaussian(sigma, norm="area")

    return convolve(vol, g, mode="constant", cval=0.0)

def apply_lut(x: np.ndarray, lo: float, hi: float, newtype=None) -> np.ndarray:
    """Clip x to the range [lo, hi], then rescale to fill the range of
    newtype."""

    if newtype is None:
        newtype = x.dtype

    y_float = (x-lo)/(hi-lo)
    y_clipped = np.clip(y_float, 0, 1)

    if np.issubdtype(newtype, np.integer):
        maxval = np.iinfo(newtype).max
    else:
        maxval = 1.0

    return (maxval*y_clipped).astype(newtype)

def auto_lut(x: np.ndarray, quantiles=(0.5,0.99), newtype=None) -> np.ndarray:
    """Linearly map the specified quantiles of x to the range of newtype."""

    lo = np.quantile(x, quantiles[0])
    hi = np.quantile(x, quantiles[1])

    return apply_lut(x, lo, hi, newtype=newtype)

def mip_x(vol:np.ndarray) -> np.ndarray:
    return np.transpose(np.max(vol, axis=2),
        (1, 0, *(range(2, np.ndim(vol)-1))))

def mip_y(vol:np.ndarray) -> np.ndarray:
    return np.max(vol, axis=1)

def mip_z(vol:np.ndarray) -> np.ndarray:
    return np.max(vol, axis=0)

def mip_threeview(vol: np.ndarray, scale=(4,1,1)) -> np.ndarray:
    """Combine 3 maximum intensity projections of a volume into a single
    2D array."""

    S = vol.shape[:3] * np.array(scale)
    output_shape = (S[1] + S[0],
                    S[2] + S[0])

    if vol.ndim == 4:
        output_shape = (*output_shape, 3)

    vol = np.repeat(vol, scale[0], axis=0)
    vol = np.repeat(vol, scale[1], axis=1)
    vol = np.repeat(vol, scale[2], axis=2)

    x = mip_x(vol)
    y = mip_y(vol)
    z = mip_z(vol)

    output = np.zeros(output_shape, dtype=vol.dtype)

    output[:S[1], :S[2]] = z
    output[:S[1], S[2]:] = x
    output[S[1]:, :S[2]] = y

    return output

def rgb_from_bw(x: np.ndarray, rgb=(1,1,1)) -> np.ndarray:
    """Convert a grayscale image to RGB."""

    return np.stack((rgb[0] * x, rgb[1] * x, rgb[2] * x), x.ndim)

def compare_red_green(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Autoscale 2D arrays x and y, then stack them with a blank blue channel
    to form a 3D array with shape (3, size_Y, size_X) and type uint8."""

    r = auto_lut(x, newtype=np.uint8)
    g = auto_lut(y, newtype=np.uint8)
    b = np.zeros_like(r)

    return np.dstack([r,g,b])

def shift(x: np.ndarray, xyz=(0,0,0), xyz_axis=(2,1,0)) -> np.ndarray:
    """Shift x in x, y and z directions (wraps numpy roll)."""

    for i in range(3):
        x = np.roll(x, xyz[i], xyz_axis[i])

    return x

def rotate_z(vol:np.ndarray, angle: float, reshape=False) -> np.ndarray:
    """Rotate the given volume about z by angle (in degrees)."""

    return rotate(vol, angle, axes=(2, 1), reshape=reshape)

def rotate_180(vol:np.ndarray, axis: int, reshape=False) -> np.ndarray:
    """Rotates the given volume about the given axis by 180 degrees by
    flipping twice."""

    if axis == 1:
        axes = (0, 2)
    elif axis == 0:
        axes = (1, 2)
    else:
        axes = (0, 1)

    return rotate(vol, 180, axes=axes, reshape=reshape)

def _coord_from_idx(idx: int, shape: int) -> float:
    return (idx + 0.5)/shape

def _idx_from_coord(coord: float, shape: int) -> int:
    return max(floor(coord*shape - 1e-6), 0)

def coords_from_idx(idx: tuple, shape: tuple) -> tuple:
    """Return image coordinates in (0, 1) for the center of a pixel located
    at index pixel_idx in [0, shape-1]. The returned coordinates cannot be 0 or
    1 because the pixel centers cannot be located at the edges of the
    volume."""
    return tuple((_coord_from_idx(i, s) for (i, s) in zip(idx, shape)))

def idx_from_coords(coords: tuple, shape: tuple) -> tuple:
    return tuple((_idx_from_coord(c, s) for (c, s) in zip(coords, shape)))

def normalized_coords_from_coords(coords: tuple, shape:tuple) -> tuple:
    """Return the normalized coordinates in [0, 1] for coordinates in
    [0, shape]. Note that a coordinate (normalized or otherwise) centered on
    pixel 0 should be slightly larger than 0."""
    return tuple([c/s for (c,s) in zip(coords, shape)])

def get_centered_subarray(
    A: np.ndarray,
    center: Tuple[int],
    radius: Tuple[int]
    ) -> np.ndarray:
    """Return a subarray centered at the given coordinate with shape
    2 * radius + 1 in each dimension.  Coordinates should be in [0, shape)."""

    padding = tuple((x, x) for x in radius)
    big_A = np.pad(A, padding)
    big_center = np.array(center) + np.array(radius)
    idx = tuple(
        slice(big_center[i]-radius[i], big_center[i]+1+radius[i])
        for i in range(len(center))
    )

    return big_A[idx]

def subpixel_max_1D(A0, A1, A2) -> float:
    """Caclulate the x-coordinate of the max for a quadratic going through the
    points (0, A0), (1, A1), and (2, A2). If A1>A0 and A1>A2, this will be a
    number in [0.5, 1.5]."""

    b = -(1.5*A0 - 2*A1 + 0.5*A2)
    a = (A0 - 2*A1 + A2)/2.0

    if a == 0:
        return 1.0
    else:
        idx = -b / (2 * a)
        if idx > 1.5 or idx < 0.5 or np.isnan(idx):
            return 1.0
        else:
            return idx

def get_nearby_max(
    A: np.ndarray,
    center: Tuple[float],
    radius: Tuple[float],
    blur_sigma: Tuple[float],
    ) -> Tuple[float]:
    """Find a nearby local maximum given an array and center, both using a
    normalized (0, 1) coordinate system. It uses 1D quadratic fitting to find
    local maxima with subpixel resolution."""

    N = len(A.shape)

    center_idx = idx_from_coords(center, A.shape)
    radius_idx = idx_from_coords(radius, A.shape)
    blur_sigma_idx = idx_from_coords(blur_sigma, A.shape)
    blur_sigma_idx = tuple([max(1, sigma) for sigma in blur_sigma_idx])

    A = blur(A, blur_sigma_idx)

    local_vol = get_centered_subarray(A, center_idx, radius_idx)

    max_indices = np.where(local_vol==np.max(local_vol))
    local_vol_max_ind = [int(np.rint(np.mean(i))) for i in max_indices]

    max_ind = (np.array(center_idx)
        - np.array(radius_idx)
        + np.array(local_vol_max_ind))

    refinement_vol = get_centered_subarray(A, max_ind, (1,)*N)
    offset = np.array((0.0,)*N)
    for i in range(N):
        idx = (1,)*i + (Ellipsis,) + (1,)*(N-i-1)
        vals = refinement_vol[idx]
        offset[i] = subpixel_max_1D(*vals) - 1

    max_ind_subpixel = max_ind + offset

    return coords_from_idx(max_ind_subpixel, A.shape)


