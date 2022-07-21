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
    pool = Pool(max(os.cpu_count()-1, 1))
    thumbnails = [pool.apply_async(
        get_thumbnail, [dataset, channel, t, scale]
    ) for t in range(shape_t)]

    d = np.zeros((shape_t, shape_t))
    for i in (tqdm(range(shape_t), desc='Calculating distances', unit='frames')
              if pbar else range(shape_t)):

        for j in range(i+1, shape_t):
            dist = dist_fn(thumbnails[i].get(), thumbnails[j].get())
            if np.isnan(dist):
                d[i, j] = 2.0
            else:
                d[i, j] = dist

    pool.close()
    pool.join()

    d_full = d + np.transpose(d)
    if save:
        np.save(str(f), d_full, allow_pickle=True)

    return d_full


def get_partial_pdists(dataset, shape_t, p_list, channel,
                       dist_fn=dist_corrcoef,
                       load=True,
                       scale=(4, 16, 16),
                       pbar=False
                       ) -> np.ndarray:
    """Return pairwise distances between shape_t frames and their parents in a dataset."""

    f = dataset / 'null.npy'
    if load:
        if channel is not None:
            f = dataset / f'pdcc_c{channel}.npy'
        else:
            f = dataset / f'pdcc.npy'
    d_full = None
    if f.is_file() and load:
        d_full = np.load(str(f), allow_pickle=True)

    print('Compiling thumbnails...')
    pool = Pool(max(os.cpu_count()-1, 1))
    thumbnails = [pool.apply_async(
        get_thumbnail, [dataset, channel, t, scale]
    ) for t in range(shape_t)]

    d_partial = np.zeros(shape_t)
    for i in (tqdm(range(shape_t), desc='Calculating distances', unit='frames')
              if pbar else range(shape_t)):

        if p_list[i] < 0:
            continue

        if d_full is not None:
            d_partial[i] = d_full[i, int(p_list[i])]
        else:
            dist = dist_fn(thumbnails[i].get(), thumbnails[int(p_list[i])].get())
            if np.isnan(dist):
                d_partial[i] = 2.0
            else:
                d_partial[i] = dist

    pool.close()
    pool.join()

    return d_partial
