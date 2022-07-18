import torch.nn as nn
import torch.nn.functional as F

from inspect import getmembers, isfunction

from . import channels
from ..utils import utils
from ..utils.utils import *


class ZephOD(nn.Module):
    def __init__(self,
                 img_shape=(17, 130, 420),
                 n_channels_in=6,
                 n_channels_out=1,
                 init_nodes=16,
                 kernel=(1, 3, 3),
                 padding=1,
                 pool_kernel=(2, 2, 2)):

        super(ZephOD, self).__init__()

        self.img_shape = img_shape
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.init_nodes = init_nodes
        self.kernel = kernel
        self.padding = padding
        self.pool_kernel = pool_kernel

        self.alpha = nn.Parameter(
            torch.ones(n_channels_in) * 0.5,
            requires_grad=True
        )
        self.gamma = nn.Parameter(
            torch.ones(n_channels_in),
            requires_grad=False
        )
        self.conv1 = conv(
            self.n_channels_in,
            self.init_nodes,
            nn.ReLU(),
            self.kernel,
            self.padding
        )
        self.conv2 = conv(
            self.init_nodes,
            self.init_nodes * 2,
            nn.ReLU(),
            self.kernel,
            self.padding
        )
        self.output_layer = conv(
            self.init_nodes * 2,
            self.n_channels_out,
            nn.Sigmoid(),
            self.kernel,
            self.padding,
        )

    def preprocess(self, vol):
        with torch.no_grad():
            n_channel = vol.shape[0]
            for c in range(n_channel):
                vol = np.append(
                    vol, np.stack(
                        [f[1](vol[c]) for f in getmembers(channels, isfunction)
                         if f not in getmembers(utils, isfunction)],
                        axis=0
                    ), axis=0
                )
            for c in range(vol.shape[0]):
                vol[c] = vol[c] / (np.max(vol[c]) + 1E-8)
        return to_tensor(vol, n_dim=5, dev=self.alpha.device)

    def forward(self, input_tensor):
        if isinstance(input_tensor, np.ndarray):
            input_tensor = self.preprocess(input_tensor)
        gamma = torch.pow(input_tensor, self.gamma.view(1, -1, 1, 1, 1))
        alpha = self.alpha.view(1, -1, 1, 1, 1) * gamma
        conv1 = self.conv1(alpha)
        conv2 = self.conv2(F.max_pool3d(conv1, self.pool_kernel))
        output_tensor = self.output_layer(conv2)
        return F.interpolate(output_tensor, input_tensor.shape[2:], mode='trilinear')


def conv(n_channels_in,
         n_channels_out,
         activation,
         kernel=(1, 3, 3),
         padding=1):

    return nn.Sequential(
        nn.Conv3d(n_channels_in,
                  n_channels_out,
                  kernel,
                  padding=padding,
                  padding_mode='zeros'),
        nn.BatchNorm3d(n_channels_out),
        activation
    )
