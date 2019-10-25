import torch
import torch.nn as nn
from reshape import Reshape


def create_network(**kwargs):
    net = []
    net.append(Reshape(shape=kwargs['input_dims']))

    net.append(nn.Conv2d(
        in_channels=kwargs['hist_len'] * kwargs['ncols'], out_channels=kwargs['n_units'][0],
        kernel_size=kwargs['filter_size'][0], stride=kwargs['filter_stride'][0], padding=1
    ))
    net.append(kwargs['nl']())

    for i in range(len(kwargs['n_units']) - 1):
        # add convolutional layer
        net.append(nn.Conv2d(
            in_channels=kwargs['n_units'][i], out_channels=kwargs['n_units'][i + 1],
            kernel_size=kwargs['filter_size'][i + 1], stride=kwargs['filter_stride'][i+1]
        ))
        net.append(kwargs['nl']())

    nel = nn.Sequential(*net).forward(torch.zeros(1, *kwargs['input_dims'])).numel()

    # reshape all feature planes into a vector per example
    net.append(Reshape(shape=[nel]))

    # fully connected layer
    net.append(nn.Linear(in_features=nel, out_features=kwargs['n_hid'][0]))
    net.append(kwargs['nl']())

    for i in range(len(kwargs['n_hid']) - 1):
        # add linear layer
        net.append(nn.Linear(in_features=kwargs['n_hid'][i], out_features=kwargs['n_hid'][i + 1]))
        net.append(kwargs['nl']())

    # add the last fully connected layer (to actions)
    net.append(nn.Linear(in_features=kwargs['n_hid'][-1], out_features=kwargs['n_actions']))

    net = nn.Sequential(*net)
    if kwargs['gpu'] >= 0:
        net.cuda()

    if kwargs['verbose'] >= 2:
        print(net)
        print('Convolutional layers flattened output size:', nel)

    return net
