import torch
def mse_loss(x, y, if_clamp=True):
    if if_clamp:
        return (((x-y).clamp(-1, 1))**2).mean()
    else:
        return ((x-y)**2).mean()
