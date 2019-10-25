import torch.nn as nn
import convnet


def create_network(**kwargs):
    kwargs['n_units']        = [32, 64, 64]
    kwargs['filter_size']    = [8, 4, 3]
    kwargs['filter_stride']  = [4, 2, 1]
    kwargs['n_hid']          = [512]
    kwargs['nl']             = nn.ReLU

    return convnet.create_network(**kwargs)
