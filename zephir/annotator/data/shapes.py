# This provides functions that create shapes for generating synthetic 3D
# shapes.
#
# Author: vivekv2@gmail.com

from typing import Tuple

import numpy as np

def get_centered_zyx_meshgrid(shape: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return a tuple (Z,Y,X) of coordinates for a 3D array generated via
    np.meshgrid. shape should be an array of odd numbers, so the center value
    of the coordinate box can be (0,0,0)."""

    shape = np.array(shape)

    bounds = (shape - 1)/2

    z = np.linspace(-bounds[0], bounds[0], shape[0])
    y = np.linspace(-bounds[1], bounds[1], shape[1])
    x = np.linspace(-bounds[2], bounds[2], shape[2])

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    return (Z, Y, X)

def gaussian(sigma:np.ndarray, shape=None, dtype=np.float32, norm="max"
) -> np.ndarray:
    """Make a 3D gaussian array density with the specified shape. norm can be
    either 'max' (largest value is set to 1.0) or 'area' (sum of values is
    1.0)."""

    sigma = np.array(sigma)

    if shape is None:
        shape = 2*sigma + 1

    (Z, Y, X) = get_centered_zyx_meshgrid(shape)

    g = np.exp(-(X**2)/(2.0*sigma[2])
               -(Y**2)/(2.0*sigma[1])
               -(Z**2)/(2.0*sigma[0]))

    if norm=="max":
        g = g / np.max(g)
    elif norm=="area":
        g = g / np.sum(g)
    else:
        raise ValueError("norm must be one of 'max' or 'area'")

    return g.astype(dtype)

def ellipsoid(r:np.ndarray, shape=None, dtype=np.float32) -> np.ndarray:
    """Make an ellipse with the specified radius."""

    r = np.array(r)
    if shape is None:
        shape = 2*r + 1

    (Z, Y, X) = get_centered_zyx_meshgrid(shape)

    D = (Z/r[0])**2 + (Y/r[1])**2 + (X/r[2])**2

    return (D < 1).astype(dtype)

def place_pattern(pattern:np.ndarray, location: np.ndarray, vol: np.ndarray
) -> None:
    """Place the pattern array in vol (mutating) centered at location. For good
    centering, pattern should have odd dimension."""

    location = np.array(location)

    s = (location - (np.array(pattern.shape) - 1)/2).astype(int)
    e = (s + pattern.shape).astype(int)
    vol[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = pattern

def place_patterns(patterns: Tuple[np.ndarray, ...],
        locations: Tuple[np.ndarray, ...], vol: np.ndarray) -> None:
    """Place each pattern in vol (mutating) centered at the corresponding
    locations."""

    for i in len(patterns):
        place_pattern(patterns[i], locations[i], vol)


