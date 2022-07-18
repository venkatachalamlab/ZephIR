import os
from multiprocessing import Pool
from skimage.transform import resize

from ..utils.io import *


def dist_corrcoef(image_1, image_2):
    """Return a distance between two images corresponding to 1 minus the
    correlation coefficient between them. This can go from 0 to 2."""

    dist = 0
    for x1, x2 in zip(image_1, image_2):
        dist += (1 - np.corrcoef(x1.ravel(), x2.ravel())[0, 1])/len(image_1)
    return dist


def get_thumbnail(dataset, channel, t, scale):
    """Return low-resolution thumbnail of data volume."""

    v = get_slice(dataset, t)
    if channel is not None:
        v = v[channel]
    elif len(v.shape) == 4:
        v = np.max(v, axis=0)
    tmg = []
    new_shape = np.array([max(1, l//s) for l, s in zip(v.shape, scale)])
    for d in range(len(v.shape)):
        mip = np.max(v, axis=d)
        tmg.append(resize(mip, np.delete(new_shape, d)))
    return tmg


def get_all_pdists(dataset, shape_t, channel,
                   dist_fn=dist_corrcoef,
                   load=True, save=True,
                   scale=(4, 16, 16),
                   pbar=False
                   ) -> np.ndarray:
    """Return all pairwise distances between the first shape_t frames in a dataset."""

    f = dataset / 'null.npy'
    if load or save:
        if channel is not None:
            f = dataset / f'pdcc_c{channel}.npy'
        else:
            f = dataset / f'pdcc.npy'
    if f.is_file() and load:
        pdcc = np.load(str(f), allow_pickle=True)
        if pdcc.shape == (shape_t, shape_t):
            return pdcc

    print('Compiling thumbnails...')
    d = np.zeros((shape_t, shape_t))
    pool = Pool(max(os.cpu_count()-1, 1))
    thumbnails = [pool.apply_async(
        get_thumbnail, [dataset, channel, t, scale]
    ) for t in range(shape_t)]
    pool.close()
    for i in (tqdm(range(shape_t), desc='Calculating distances', unit='frames')
              if pbar else range(shape_t)):
        for j in range(i+1, shape_t):
            dist = dist_fn(thumbnails[i].get(), thumbnails[j].get())
            if np.isnan(dist):
                d[i, j] = 1.0
            else:
                d[i, j] = dist
    pool.join()

    d_full = d + np.transpose(d)
    if save:
        np.save(str(f), d_full, allow_pickle=True)

    return d_full
