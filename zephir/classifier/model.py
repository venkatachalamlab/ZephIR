import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronClassifier(nn.Module):
    def __init__(self,
                 img_shape=(5, 64, 64),
                 n_channels_in=3,
                 n_channels_out=1,
                 init_nodes=16,
                 lin_channels_in=7):
        super(NeuronClassifier, self).__init__()

        padding = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 1)]
        dilation = [1, 1, 1, 1]
        kernel_size = [(1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 2, 2)]
        stride = [(1, 1, 1), (1, 1, 1),  (1, 1, 1), (1, 2, 2)]


        # convolutional network for analyzing crop volumes
        self.convolution = nn.Sequential(
            nn.Conv3d(n_channels_in,
                      init_nodes,
                      kernel_size=kernel_size[0],
                      padding=padding[0],
                      stride=stride[0],
                      dilation=dilation[0],
                      padding_mode='zeros'),
            nn.BatchNorm3d(init_nodes),
            nn.ReLU(),
            nn.Conv3d(init_nodes,
                      init_nodes * 2,
                      kernel_size=kernel_size[1],
                      padding=padding[1],
                      stride=stride[1],
                      dilation=dilation[1],
                      padding_mode='zeros'),
            nn.BatchNorm3d(init_nodes * 2),
            nn.ReLU(),
            nn.Conv3d(init_nodes * 2,
                      init_nodes * 4,
                      kernel_size=kernel_size[2],
                      padding=padding[2],
                      stride=stride[2],
                      dilation=dilation[2],
                      padding_mode='zeros'),
            nn.BatchNorm3d(init_nodes * 4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=kernel_size[3],
                         padding=padding[3],
                         stride=stride[3],
                         dilation=dilation[3]),

            # additional layers go here
        )

        self.output_img_shape = [img_shape[0], img_shape[1], img_shape[2]]
        for i in range(len(padding)):
            for j in range(3):
                self.output_img_shape[j] = int(
                    (self.output_img_shape[j] + 2 * padding[i][j] - dilation[i] * (kernel_size[i][j] - 1) - 1) /
                    stride[i][j] + 1)
        # print(img_shape, self.output_img_shape)
        flattendConvOut = int(torch.prod(torch.tensor([*self.output_img_shape, init_nodes * 4])))
        flattened_size = (
                             # number of pixels in convolution output * number of latent channels
                                 flattendConvOut
                                 # + size of output from last linear layer
                                 + init_nodes * 3
        )
        flattened_size = init_nodes*6
        self.convTransform = nn.Sequential(
            nn.Linear(flattendConvOut, init_nodes),
            nn.ReLU(),
            nn.Linear(init_nodes, init_nodes * 2),
            nn.ReLU(),
            nn.Linear(init_nodes * 2, init_nodes * 3),
            nn.ReLU(),
        )

        # linear network for analyzing coordinates, etc.
        self.linear = nn.Sequential(
            nn.Linear(lin_channels_in, init_nodes),
            nn.ReLU(),
            nn.Linear(init_nodes, init_nodes * 2),
            nn.ReLU(),
            nn.Linear(init_nodes * 2, init_nodes * 3),
            nn.ReLU(),
            # additional layers go here
        )

        # linear classifier
        self.output_layer = nn.Sequential(
            nn.Linear(flattened_size, init_nodes),
            nn.ReLU(),
            nn.Linear(init_nodes, init_nodes*2),
            nn.ReLU(),
            # additional layers go here
            nn.Linear(init_nodes*2, init_nodes * 3),
            nn.ReLU(),
            nn.Linear(init_nodes*3, n_channels_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_volume, input_coordinates, weights=None):
        conv = self.convolution(input_volume)
        convTransform = self.convTransform(conv.view(conv.size(0), -1))
        if weights is not None:
            linear_partial = self.linear[1:]

            linear = linear_partial(F.linear(input_coordinates, weight=weights))
        else:
            linear = self.linear(input_coordinates)

        # output_tensor = self.output_layer(linear)
        # output_tensor = self.output_layer(
        #     torch.cat(
        #         (conv.view(conv.size(0), -1),
        #          linear), dim=-1
        #     )
        output_tensor = self.output_layer(torch.cat((convTransform, linear), dim=-1))

        return output_tensor