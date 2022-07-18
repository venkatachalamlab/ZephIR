import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from ..utils.utils import get_gaussian_kernel


class ZephIR(nn.Module):
    def __init__(self,
                 allow_rotation,
                 dimmer_ratio,
                 fovea_sigma,
                 grid_shape,
                 grid_spacing,
                 n_chunks,
                 n_frame,
                 shape_n,
                 ftr_ratio=0.6,
                 ret_stride=2,):

        super(ZephIR, self).__init__()

        self.n_chunks = max(n_chunks, 1)
        self.n_frame = n_frame
        self.shape_n = shape_n

        # calculating descriptor sampling stride
        # foveated region at the center has full resolution (stride=1)
        # retinal region at the edges has reduced resolution (stride=ret_stride)
        grid_fovea = []
        dimmer = []
        for dim, reach in zip(grid_shape, grid_spacing):
            fovea_idx = int((1 - ftr_ratio) * dim / 2)
            retina = torch.zeros(dim)
            dimmer_ = torch.ones(dim)
            if fovea_idx > 0:
                for i in range(fovea_idx + 1):
                    retina[fovea_idx - i] += -(ret_stride - 1) * (reach * 2 / dim) * i
                    retina[-(fovea_idx - i + 1)] += (ret_stride - 1) * (reach * 2 / dim) * i
                    dimmer_[fovea_idx - i] *= dimmer_ratio
                    dimmer_[-(fovea_idx - i + 1)] *= dimmer_ratio
            grid_fovea.append(retina)
            dimmer.append(dimmer_)
        x_dimmer, y_dimmer, z_dimmer = torch.meshgrid(dimmer, indexing='ij')

        # calculating Gaussian mask to modulate intensity
        # center has full intensity, with Gaussian roll-off towards the edges
        # the multiplier never drops below dimmer_ratio
        if fovea_sigma[-1] < 0.:
            fovea_kernel = torch.ones(grid_shape)
        else:
            fovea_kernel = get_gaussian_kernel(grid_shape, fovea_sigma, channels=0)
            fovea_kernel = fovea_kernel / torch.max(fovea_kernel)

        # consolidates foveation effects
        self.dimmer = nn.Parameter(
            torch.clamp(
                torch.clamp(x_dimmer*y_dimmer*z_dimmer, dimmer_ratio, 1.0) * fovea_kernel,
                dimmer_ratio, 1.0
            ), requires_grad=False
        )

        # pre-compiling an identity sampling grid
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, grid_shape[0]) * grid_spacing[0] + grid_fovea[0],
            torch.linspace(-1, 1, grid_shape[1]) * grid_spacing[1] + grid_fovea[1],
            torch.linspace(-1, 1, grid_shape[2]) * grid_spacing[2] + grid_fovea[2],
            indexing='ij'
        )
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1)
        self.grid = nn.Parameter(
            grid.expand(self.n_frame, self.shape_n, -1, -1, -1, -1),
            requires_grad=False
        )

        # transformation parameter
        # direct correspondence to keypoint coordinates
        self.rho = nn.Parameter(
            torch.zeros((self.n_frame, self.shape_n, 3)),
            requires_grad=True
        )

        # transformation parameter
        # controls rotation of descriptors about center in the xy-plane
        self.theta = nn.Parameter(
            torch.zeros((self.n_frame, self.shape_n, 1)),
            requires_grad=allow_rotation
        )

    def forward(self, input_tensor):

        # applying rotation
        s_theta = torch.sin(self.theta).view(self.n_frame, self.shape_n, 1, 1, 1)
        cs_theta = torch.cos(self.theta).view(self.n_frame, self.shape_n, 1, 1, 1)
        theta = torch.stack((
            self.grid[..., 0] * cs_theta - self.grid[..., 1] * s_theta,
            self.grid[..., 0] * s_theta + self.grid[..., 1] * cs_theta,
            self.grid[..., 2]
        ), dim=-1)

        output_list = []
        for i in range(input_tensor.shape[0]):

            # forward pass is divided into n_chunks to reduce memory consumption
            for _theta, _rho in \
                zip(torch.chunk(theta[i], self.n_chunks, dim=0),
                    torch.chunk(self.rho[i], self.n_chunks, dim=0)):

                output_list.append(
                    grid_sample(
                        input_tensor[i].expand(_rho.shape[0], -1, -1, -1, -1),
                        _theta + _rho.view(-1, 1, 1, 1, 3),         # applying translation
                        align_corners=False, padding_mode='border'
                    )
                )

        output_tensor = torch.cat(output_list, dim=0).view(
            self.n_frame, self.shape_n,
            input_tensor.shape[2], *self.dimmer.shape
        )
        return output_tensor * self.dimmer.expand_as(output_tensor)
