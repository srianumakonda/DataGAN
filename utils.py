import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def visualize_images(batch, filepath=None, save_image=False, single=False):
    grid = torchvision.utils.make_grid(batch, nrow=8)
    if single:
        grid = torchvision.utils.make_grid(batch, nrow=1)
    if save_image:
        torchvision.utils.save_image(grid, filepath)
    else:
        plt.figure(figsize=(50,50))
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')