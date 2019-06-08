import torch
import torch.nn as nn
import torch.nn.functional as functional

class BasicMagnifyCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, magnify_factor=1, pooling_size=2, if_Bn=True, activation=nn.ReLU(inplace=True)):
        super(BasicMagnifyCell, self).__init__()
        self.mag_factor = magnify_factor
        self.downSampling = torch.nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.if_Bn = if_Bn
        if self.if_Bn:
            self.Bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
        
    def forward(self, x):
        down_x = self.downSampling(x)
        coarse_grain = functional.interpolate(down_x, size=x.shape[2:4], mode="bilinear", align_corners=True)
        fine_grain = x - coarse_grain
        mag_x = x + fine_grain * self.mag_factor
        output = self.conv2d(mag_x)
        if self.if_Bn:
            output = self.Bn(output)
        output = self.activation(output)
        return output