import colorsys
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .io import *


# core utilities for ZephIR
def blur3d(input_tensor, kernel_size, sigmas, dev=torch.device('cpu')):
    return F.conv3d(
        input_tensor.view(-1, *input_tensor.shape[2:]),
        get_gaussian_kernel(
            kernel_size=kernel_size,
            sigmas=sigmas,
            channels=input_tensor.size(2),
            dev=dev
        ),
        groups=input_tensor.size(2),
        padding=tuple([int((size - 1) / 2) for size in kernel_size])
    ).view(input_tensor.shape)


def get_gaussian_kernel(kernel_size=(1, 3, 3), sigmas=(1, 2, 2), channels=2, dev=torch.device('cpu')):
    kernel = torch.ones(kernel_size, device=dev)
    meshgrids = torch.meshgrid(
        [torch.arange(size, device=dev) for size in kernel_size],
        indexing='ij'
    )
    for size, std, mgrid in zip(kernel_size, sigmas, meshgrids):
        mean = (size - 1) / 2
        kernel *= (1 / (std * np.sqrt(2 * np.pi))
                   * torch.exp(-torch.pow((mgrid - mean) / std, 2) / 2))

    # kernel must sum to one
    kernel = kernel / torch.sum(kernel)

    # repeat over channels
    if channels > 0:
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, 1, *[1] * len(kernel_size))

    return kernel


def to_tensor(data, n_dim=None, grad=False, dtype=None, dev=torch.device('cpu')):
    if isinstance(data, np.ndarray) or isinstance(data, list):
        if n_dim is not None:
            while len(data.shape) < n_dim:
                data = np.array(data)[np.newaxis, ...]
        if dtype is not None:
            return torch.tensor(data, requires_grad=grad, device=dev, dtype=dtype)
        return torch.tensor(data, requires_grad=grad, device=dev, dtype=torch.float32)
    else:
        return data


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.clone().cpu().detach().numpy()
    else:
        return tensor


def get_pixel(coords, img_shape):
    if isinstance(coords, torch.Tensor):
        return idx_from_coords(
            tuple((to_numpy(coords) + 1) / 2),
            tuple(np.array(img_shape)[::-1])
        )
    else:
        return idx_from_coords(
            tuple((coords + 1) / 2),
            tuple(np.array(img_shape)[::-1])
        )


def pick_color(x):
    return 255 * np.array(
        colorsys.hls_to_rgb(
            ((x * 157) % 360) / 360,
            (65 + (x * 7) % 35) / 100,
            (47 + (x * 7) % 53) / 100
        )
    )


def save_as_bgr(img, scale=(2, 1, 1), annotation=None, path=None, s=2):
    if np.max(img) <= 1.0:
        img *= 255

    composite = []
    for channel in range(img.shape[0]):
        if np.max(img[channel, ...]) > 1e-8:
            composite.append(
                    mip_threeview(
                            np.uint8(img[channel, ...]),
                            scale
                    )
            )

    if len(composite) == 3:
        r = auto_lut(composite[0], newtype=np.uint8)
        g = auto_lut(composite[1], newtype=np.uint8)
        b = auto_lut(composite[2], newtype=np.uint8)
        frame = np.dstack([r, g, b])
    elif len(composite) == 2:
        frame = compare_red_green(composite[0], composite[1])
    elif len(composite) == 1:
        r = auto_lut(composite[0], newtype=np.uint8)
        g = np.zeros_like(r)
        b = np.zeros_like(r)
        frame = np.dstack([r, g, b])
    else:
        r = mip_threeview(np.uint8(img[0, ...]), scale)
        g = np.zeros_like(r)
        b = np.zeros_like(r)
        frame = np.dstack([r, g, b])

    if annotation is not None:
        for n in range(annotation.shape[0]):
            scaled_shape = img.shape[1:] * np.array(scale)
            x, y, z = get_pixel(annotation[n, ...], scaled_shape)
            color = pick_color(n)
            frame = cv2.circle(frame, (int(x), int(y)), s, color=color, thickness=-1)
            frame = cv2.circle(frame, (int(x), int(z + scaled_shape[1])), s, color=color, thickness=-1)
            frame = cv2.circle(frame, (int(z + scaled_shape[2]), int(y)), s, color=color, thickness=-1)

    if path is not None:
        cv2.imwrite(str(path), frame[:, :, ::-1])

    return frame


def live_evolve_neuron(neurons, msg):
    from IPython import display
    _ = display.clear_output(wait=True)
    _ = plt.cla()
    _ = plt.clf()
    f, axes = plt.subplots(1, len(neurons))
    for n, descriptor in enumerate(neurons):
        if descriptor.shape[0] == 2:
            descriptor = np.append(descriptor, np.zeros((1, *descriptor.shape[1:])), axis=0)
        elif descriptor.shape[0] == 1:
            descriptor = np.append(descriptor, np.zeros((2, *descriptor.shape[1:])), axis=0)
        _ = axes[n].imshow(np.max(np.transpose(descriptor, (1, 2, 3, 0)), axis=0) / np.max(descriptor))
    print(msg)
    _ = display.display(plt.gcf())


def plot_with_indicator_v(dataset, indicators, x_list=None, title=''):
    color_list = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(len(indicators)):
        pt, start, end = indicators[i]
        color = color_list[i % len(color_list)]
        plt.vlines(pt, start, end, color=color)
    for j in range(len(dataset)):
        if x_list is None:
            plt.plot(dataset[j])
        else:
            plt.plot(x_list[j], dataset[j])
    plt.title(title)
    plt.show()


# ported over from vlab.images.transform
def _idx_from_coords(coords: float, shape: int) -> int:
    return max(round(coords*shape - 1E-6), 0)


def idx_from_coords(coords: tuple, shape: tuple) -> tuple:
    return tuple((_idx_from_coords(c, s) for (c, s) in zip(coords, shape)))


def _coords_from_idx(idx: int, shape: int) -> float:
    return (idx + 0.5) / shape


def coords_from_idx(idx: tuple, shape: tuple) -> tuple:
    return tuple((_coords_from_idx(i, s) for (i, s) in zip(idx, shape)))


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


def mip_threeview(vol: np.ndarray, scale=(4,1,1)) -> np.ndarray:
    """Combine 3 maximum intensity projections of a volume into a single
    2D array."""

    s = vol.shape[:3] * np.array(scale)
    output_shape = (s[1] + s[0],
                    s[2] + s[0])

    if vol.ndim == 4:
        output_shape = (*output_shape, 3)

    vol = np.repeat(vol, scale[0], axis=0)
    vol = np.repeat(vol, scale[1], axis=1)
    vol = np.repeat(vol, scale[2], axis=2)

    x = np.transpose(np.max(vol, axis=2), (1, 0, *(range(2, np.ndim(vol)-1))))
    y = np.max(vol, axis=1)
    z = np.max(vol, axis=0)

    output = np.zeros(output_shape, dtype=vol.dtype)

    output[:s[1], :s[2]] = z
    output[:s[1], s[2]:] = x
    output[s[1]:, :s[2]] = y

    return output


def compare_red_green(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Autoscale 2D arrays x and y, then stack them with a blank blue channel
    to form a 3D array with shape (3, size_Y, size_X) and type uint8."""

    r = auto_lut(x, newtype=np.uint8)
    g = auto_lut(y, newtype=np.uint8)
    b = np.zeros_like(r)

    return np.dstack([r, g, b])


def fill_nans(data):
    temp = data.copy()
    nans_idx = np.where(np.isnan(temp))[0]
    for i in nans_idx:
        j = i - 1
        while data[j] == np.nan:
            j += -1
        temp[i] = temp[j]
    return temp


# ported from vlab.images.order_times
def get_undiscounted_scores_for_tree(p_dist, parents, shape_t):
    scores = np.zeros(shape_t)
    for i in range(shape_t):
        if parents[i] < 0:
            continue
        scores[i] = p_dist[i, int(parents[i])]
    return scores


def get_depth(parents, node, node_depth=0):
    """Get depth of a given node given a tree formated as a list of parents,
    with the root pointing to -1."""

    parent = parents[node]
    if parent == -1:
        return node_depth
    else:
        return get_depth(parents, parent, node_depth + 1)


# utilities for Zephod
def gaussian(sigma: np.ndarray, shape=None, dtype=np.float32, norm="max"
             ) -> np.ndarray:
    """Make a 3D gaussian array density with the specified shape. norm can be
    either 'max' (largest value is set to 1.0) or 'area' (sum of values is
    1.0)."""

    sigma = np.array(sigma)
    if shape is None:
        shape = 2*sigma + 1
    shape = np.array(shape).astype(int)
    bounds = ((shape - 1)/2).astype(int)

    z = np.linspace(-bounds[0], bounds[0], shape[0])
    y = np.linspace(-bounds[1], bounds[1], shape[1])
    x = np.linspace(-bounds[2], bounds[2], shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    g = np.exp(-(X**2)/(2.0*sigma[2]) - (Y**2)/(2.0*sigma[1]) - (Z**2)/(2.0*sigma[0]))

    if norm == "max":
        return (g / np.max(g)).astype(dtype)
    elif norm == "area":
        return (g / np.sum(g)).astype(dtype)
    else:
        raise ValueError("norm must be one of 'max' or 'area'")


def get_times(dataset_path: Path) -> np.ndarray:
    """Return time indices in a dataframe.
    This should return a 1-D numpy array containing integer indices for get_slice().
    """
    h5_filename = dataset_path / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return np.arange(len(f["times"][:]))
