"""
ZephOD network uses all functions listed below as input channels to combine/select from.
This file may be edited by a user to fit their particular use case,
or to add new State-of-the-Art segmentation/detection algorithms,
but each edit will require retraining of the weights.
Each function needs to take only the volumetric data slice (Z, Y, X) as the input,
and return the processed volume with the same shape as the output.
Each function will be iterated over all data channels.
"""

import cv2
import numpy as np
from skimage import restoration

from ..utils.utils import gaussian


def rl_deconvolution(img, sigma=5, gamma=1.0, lam=12, n_iter=10):
    deconvolved = np.zeros_like(img)
    kernel = gaussian(
        np.array([1, sigma, sigma]),
        shape=None, dtype=np.float32, norm="max"
    )
    for z in range(img.shape[0]):
        gray = (img[z].astype('float64') / (np.max(img[z]) + 1E-8)) ** (1 / gamma)
        image_noisy = gray + (np.random.poisson(lam=lam, size=gray.shape) - 10) / 255
        deconvolved[z] = restoration.richardson_lucy(image_noisy, kernel[1], iterations=n_iter)

    return deconvolved * 255


def apply_255_lut(img: np.ndarray, lo=0, hi=255) -> np.ndarray:
    if lo == 0:
        lo = max(np.amin(img), 0.1 * np.amax(img))
    if hi == 255:
        hi = np.amax(img)

    newtype = img.dtype

    y_float = (img - lo) / (hi - lo)
    y_clipped = np.clip(y_float, 0, 255)

    if np.issubdtype(newtype, np.integer):
        maxval = np.iinfo(newtype).max
    else:
        maxval = 255.0

    return (maxval * y_clipped).astype(newtype)


# Img must be of shape - (Z,Y,X) (channels not included)
def threshold(img: np.ndarray, lo=50, hi=200) -> np.ndarray:
    ret, thresh = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY)
    kernel = np.ones((1, 2), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    return erosion


# Img must be of shape - (Z,Y,X) (channels not included)
def sharpen(img: np.ndarray,
            sharp=np.array([[0, -3, 0], [-3, 16, -3], [0, -3, 0]])
            ) -> np.ndarray:
    img_lut = apply_255_lut(img)
    sharpen_img = cv2.filter2D(img_lut, -1, sharp)
    return sharpen_img
