"""
This file collects all functions directly related to calculating
the four major components of ZephIR's loss function:
1. corr_loss: normalized correlation loss for image registration
2. reg_n: spring network maintaining flexible connections to nearest neighbors
3. reg_t: linear temporal smoothing across a small sequence of frames
4. reg_d: nuclei probability as detected by ZephOD network
"""
import torch
from torch.nn.functional import grid_sample

from ..utils.utils import *


def corr_loss(prediction, target):
    """Image registration loss, L_R.

    Normalized correlation loss between two lists of volumes (T, C, Z, Y, X).
    Loss is calculated over (Z, Y, X) axes and averaged over (C) axis. (T) axis
    is not reduced.

    :param prediction: child descriptors
    :param target: target descriptors
    :return: loss
    """

    vx = prediction - torch.mean(prediction, dim=[2, 3, 4, 5], keepdim=True)
    vy = target - torch.mean(target, dim=[2, 3, 4, 5], keepdim=True)
    # child descriptors can sometimes be empty and cause DivByZero error
    if torch.any(torch.std(prediction, dim=[2, 3, 4, 5]) == 0):
        sxy = torch.mul(torch.std(target, dim=[2, 3, 4, 5]),
                        torch.std(target, dim=[2, 3, 4, 5]))
    else:
        sxy = torch.mul(torch.std(prediction, dim=[2, 3, 4, 5]),
                        torch.std(target, dim=[2, 3, 4, 5]))
    cc = torch.div(torch.mean(torch.mul(vx, vy), dim=[2, 3, 4, 5]), sxy + 1e-5)
    return torch.mean(1 - cc, dim=1)


def reg_n(k, rho, rho_p, ind, d_ref, subset=None, mode='disp'):
    """Spatial regularization, L_N.

    Spring-like cost to deformations in a network of nearest-neighbor connections,
    penalizing relative motion between keypoints and their neighbors, calculated
    according to given mode:
    'disp': linear cost to all relative motion
    'norm': linear cost to change in distance between keypoints, rotation is allowed
    'ljp': Lennard-Jones cost to change in distances between keypoints,
    rotation is allowed, collapsing onto the same coordinates is heavily penalized

    :param k: spring stiffnesses
    :param rho: keypoint coordinates
    :param rho_p: restricted keypoint coordinates
    :param ind: neighbor indices
    :param d_ref: neighbor displacements in reference frame
    :param subset: indices of keypoints in restricted list
    :param mode: method for calculating loss
    :return: loss
    """

    if rho_p is not None and subset is not None:
        nn_rho = torch.cat([rho_p[:, ind[:, n], :].unsqueeze(2)
                              for n in range(ind.shape[1])], dim=2)
        d = nn_rho[:, subset, ...] - rho.unsqueeze(2).expand(-1, -1, ind.shape[1], -1)
        d_ref = d_ref[subset, ...]
        k = k[subset, ...]
    else:
        nn_rho = torch.cat([rho[:, ind[:, n], :].unsqueeze(2)
                              for n in range(ind.shape[1])], dim=2)
        d = nn_rho - rho.unsqueeze(2).expand(-1, -1, ind.shape[1], -1)

    if mode == 'norm':
        d_ref = torch.norm(d_ref, dim=-1).unsqueeze(0)
        d_diff = torch.div(torch.abs(torch.norm(d, dim=-1) - d_ref), d_ref + 1e-5)
    elif mode == 'ljp':
        d_ref = torch.norm(d_ref, dim=-1).unsqueeze(0)
        d_ljp = torch.div(d_ref, torch.norm(d + 1e-5, dim=-1))
        d_diff = torch.pow(d_ljp, 4) - 2 * torch.pow(d_ljp, 2)
    else:
        d_diff = torch.div(torch.norm(d - d_ref.unsqueeze(0), dim=-1),
                           torch.norm(d_ref + 1e-5, dim=-1).unsqueeze(0))

    reg = torch.mean(k.unsqueeze(0) * d_diff, dim=-1)
    reg = [r[torch.isfinite(r)] for r in reg]
    reg = [torch.mean(r) if len(r) > 0 else torch.tensor(0).to(r.dtype) for r in reg]
    return torch.stack(reg, dim=0)


def reg_t(k, descriptors, crop_rads, npx_to_keep):
    """Temporal smoothing, L_T.

    Extracts pixel intensities at center of descriptors and penalizes change
    in intensity from the center frame.

    :param k: multiplier to modulate relative contribution to loss, lambda_T
    :param descriptors: child descriptors (T, C, Z, Y, X)
    :param crop_rads: radius of center crop (Z, Y, X)
    :param npx_to_keep: number of pixels to keep when calculating average intensity value
    :return: loss
    """

    shape_t, shape_n, shape_c = descriptors.shape[:3]
    traces = torch.zeros(shape_t, shape_n, shape_c)

    shape_zyx = np.array([descriptors.shape[3], descriptors.shape[4], descriptors.shape[5]])
    crop_i, crop_f = shape_zyx // 2 - crop_rads, shape_zyx // 2 + crop_rads + 1
    descriptors = descriptors[..., crop_i[0]:crop_f[0], :, :]
    descriptors = descriptors[..., crop_i[1]:crop_f[1], :]
    descriptors = descriptors[..., crop_i[2]:crop_f[2]]
    descriptors = descriptors.reshape(shape_t, shape_n, shape_c, -1)

    for t in range(shape_t):
        for n in range(shape_n):
            for c in range(shape_c):
                if npx_to_keep is not None:
                    roi = torch.sort(descriptors[t, n, c])[0][-npx_to_keep:]
                else:
                    roi = torch.sort(descriptors[t, n, c])[0]
                traces[t, n, c] = torch.mean(roi)

    traces = torch.abs(traces - traces[shape_t // 2, ...].unsqueeze(0))
    return k * torch.mean(torch.sum(traces, dim=-1), dim=-1)


def reg_d(k, pred, rho, n_chunks):
    """Feature detection loss, L_D.

    Calculates feature detection probability at keypoint coordinates.

    :param k: multiplier to modulate relative contribution to loss, lambda_D
    :param pred: Zephod prediction of feature detection probability map
    :param rho: keypoint coordinates
    :param n_chunks: number of chunks to divide forward process into
    :return: loss
    """
    shape_t, shape_n = rho.shape[:2]
    reg = []
    for t in range(shape_t):
        for _rho in torch.chunk(rho[t], n_chunks, dim=0):
            reg.append(grid_sample(
                pred[t].expand(_rho.shape[0], -1, -1, -1, -1),
                _rho.view(-1, 1, 1, 1, 3),
                align_corners=False, padding_mode='border'
            ))
    reg = torch.cat(reg, dim=0).view((shape_t, shape_n))
    return k * torch.mean(reg, dim=-1)
