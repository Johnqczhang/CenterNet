import torch.nn as nn

from detectron2.layers import Conv2d
from detectron2.layers.deform_conv import ModulatedDeformConv
from detectron2.layers.deform_conv import modulated_deform_conv


class DCN(ModulatedDeformConv):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            deformable_groups,
        )

        offset_base_channels = kernel_size * kernel_size
        offset_channels = offset_base_channels * 3  # default: 27

        self.conv_offset_mask = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation
        )
        nn.init.kaiming_uniform_(self.conv_offset_mask.weight, a=1)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        offset = offset_mask[:, :self.offset_split, :, :]
        mask = offset_mask[:, self.offset_split:, :, :].sigmoid()
        return modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )
