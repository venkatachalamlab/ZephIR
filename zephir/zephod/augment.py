import cv2
import random

from inspect import getmembers, isfunction
from scipy.ndimage import rotate
# from stardist.models import StarDist2D

from . import channels
from ..utils import utils
from ..utils.utils import *


def identify(vol, annotations, r_crop):
    isolates = []

    for annot in annotations:
        x, y, z = get_pixel(annot, vol.shape[1:])
        slice_yx = np.max(vol[0, max(0, z-1):z+1, y-r_crop:y+r_crop, x-r_crop:x+r_crop], axis=0)
        slice_zy = np.max(vol[0, max(0, z-2):z+2, y-r_crop:y+r_crop, x-1:x+1], axis=2)
        # contours_yx = measure.find_contours(slice_yx, 0.8)
        # contours_zy = measure.find_contours(slice_zy, 0.8)
        contours_yx = cv2.findContours(slice_yx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours_zy = cv2.findContours(slice_zy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(contours_yx) != 1 or len(contours_zy) != 1:
            continue

        crop = vol[:, max(0, z-2):z+2, y-r_crop:y+r_crop, x-r_crop:x+r_crop].copy()
        for channel in range(crop.shape[0]):
            for zslice in range(crop.shape[1]):
                if channel == 0:
                    _, thresh = cv2.threshold(crop[channel, zslice], 30, 255, cv2.THRESH_TOZERO)
                else:
                    _, thresh = cv2.threshold(crop[channel, zslice], 30, 255, cv2.THRESH_TOZERO)
                cntr = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                mask = np.zeros_like(crop[channel, zslice])
                for c in cntr:
                    if cv2.pointPolygonTest(c, (mask.shape[0]//2, mask.shape[1]//2), False) > 0:
                        mask = cv2.drawContours(mask, [c], -1, 1, -1)
                        mask = cv2.drawContours(mask, [c], -1, 1, 3)
                crop[channel, zslice] = crop[channel, zslice] * mask

        if np.max(crop) > 0:
            isolates.append(crop)
            # plt.imshow(np.max(crop[0], axis=0))
            # plt.show()
    return isolates


def generate_synthetic_data(vol, annotations, isolates, preprocess=False):
    synthetic = np.zeros_like(vol)
    labels = np.zeros((1, *vol.shape[1:]))
    for n, annot in enumerate(annotations):
        x, y, z = get_pixel(annot, vol.shape[1:])
        neuron = isolates[random.randint(0, len(isolates)-1)]

        rotate_idx = random.randint(0, 1)
        if rotate_idx == 1:
            neuron = rotate_neuron(neuron)

        resize_idx = random.randint(0, 1)
        if resize_idx == 1:
            neuron = resize_neuron(neuron)

        reposition_idx = random.randint(0, 1)
        if reposition_idx == 1:
            x += random.randrange(-5, 5)
            y += random.randrange(-5, 5)
        #     z += random.randrange(-1, 1)

        while z-neuron.shape[1]//2 < 0:
            z += 1
        while z-neuron.shape[1]//2+neuron.shape[1] > synthetic.shape[1]:
            z += -1
        while y-neuron.shape[2]//2 < 0:
            y += 1
        while y-neuron.shape[2]//2+neuron.shape[2] > synthetic.shape[2]:
            y += -1
        while x-neuron.shape[3]//2 < 0:
            x += 1
        while x-neuron.shape[3]//2+neuron.shape[3] > synthetic.shape[3]:
            x += -1

        synthetic[
            :, z-neuron.shape[1]//2:z-neuron.shape[1]//2+neuron.shape[1],
            y-neuron.shape[2]//2:y-neuron.shape[2]//2+neuron.shape[2],
            x-neuron.shape[3]//2:x-neuron.shape[3]//2+neuron.shape[3]
        ] = np.max(
            np.append(
                synthetic[
                    np.newaxis, :,
                    z-neuron.shape[1]//2:z-neuron.shape[1]//2+neuron.shape[1],
                    y-neuron.shape[2]//2:y-neuron.shape[2]//2+neuron.shape[2],
                    x-neuron.shape[3]//2:x-neuron.shape[3]//2+neuron.shape[3]
                ], neuron[np.newaxis, ...],
                axis=0
            ),
            axis=0
        )

        labels[:, z, y, x] = 1

    flip_idx = random.randint(-1, 2)
    if flip_idx < 2:
        for z in range(synthetic.shape[1]):
            for c in range(synthetic.shape[0]):
                synthetic[c][z] = cv2.flip(synthetic[c][z], flip_idx)
            labels[0][z] = cv2.flip(labels[0][z], flip_idx)

    if preprocess:
        for c in range(vol.shape[0]):
            synthetic = np.append(
                synthetic, np.stack(
                    [f[1](synthetic[c]) for f in getmembers(channels, isfunction)
                     if f not in getmembers(utils, isfunction)],
                    axis=0
                ), axis=0
            )
        for c in range(synthetic.shape[0]):
            synthetic[c] = synthetic[c] / (np.max(synthetic[c]) + 1E-8)

    return synthetic, labels


def rotate_neuron(neuron):
    angle = random.randrange(-40, 40)
    neuron = rotate(
        neuron, angle,
        axes=(3, 2), reshape=True, output=None,
        order=3, mode='constant', cval=0.0, prefilter=True
    )
    return neuron


def resize_neuron(neuron, shape=None):
    if shape is None:
        size = random.randrange(-3, 5)
        shape = (max(3, neuron.shape[2]+size), max(3, neuron.shape[3]+size))
    resized = np.zeros((*neuron.shape[:2], *shape))
    for channel in range(neuron.shape[0]):
        for zslice in range(neuron.shape[1]):
            resized[channel, zslice] = resize(
                neuron[channel, zslice],
                shape
            )
    return (resized * 255).astype(neuron.dtype)
