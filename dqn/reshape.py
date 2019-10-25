import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = [-1] + shape

    def forward(self, x):
        return torch.reshape(input=x, shape=self.shape).contiguous()
