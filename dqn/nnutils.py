import torch


def abs_mean(w):
    return w.abs().mean().detach().cpu().tolist()


def abs_max(w):
    return w.abs().max().detach().cpu().tolist()


# Build a string of average absolute weight values for the modules in the
# given network.
def get_weight_norms(module):
    s = '\n'
    title = 'Weight Norms'
    s += '-' * ((30 - len(title)) // 2) + title + '-' * ((30 - len(title) + 1) // 2) + '\n'
    for name, param in module.named_parameters():
        s += '%-12s   %15.12f\n' % (name, abs_mean(param))
    s += '\n'
    title = 'Weight Max'
    s += '-' * ((30 - len(title)) // 2) + title + '-' * ((30 - len(title) + 1) // 2) + '\n'
    for name, param in module.named_parameters():
        s += '%-12s   %15.12f\n' % (name, abs_max(param))
    return s


# Build a string of average absolute weight gradient values for the modules
# in the given network.
def get_grad_norms(module):
    s = '\n'
    title = 'Weight Grad Norms'
    s += '-' * ((30 - len(title)) // 2) + title + '-' * ((30 - len(title) + 1) // 2) + '\n'
    for name, param in module.named_parameters():
        if param.grad is not None:
            s += '%-12s   %15.12f\n' % (name, abs_mean(param.grad))
    s += '\n'
    title = 'Weight Grad Max'
    s += '-' * ((30 - len(title)) // 2) + title + '-' * ((30 - len(title) + 1) // 2) + '\n'
    for name, param in module.named_parameters():
        if param.grad is not None:
            s += '%-12s   %15.12f\n' % (name, abs_max(param.grad))
    return s
