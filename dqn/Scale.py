import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb2y(x):
    assert x.shape[-1] == 3
    return torch.matmul(x, torch.tensor([0.299, 0.587, 0.114]))


class Scale(nn.Module):
    def __init__(self, height, width):
        super(Scale, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x):
        if x.dim() > 3:
            print('Scale.py: WARNING: `x.dim() > 3`')
            x = x[0]

        assert x.dtype == torch.float32
        x = rgb2y(x)
        x = x.view([1, 1, 210, 160])
        x = F.interpolate(x, size=(self.width, self.height), mode='bilinear')
        x = x.view([1, 84, 84])
        return x
