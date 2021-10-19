import torch
import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.discrim = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 256),
            self._block(256, 256),
            self._block(256, 256),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.discrim(x)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))

class Generator(nn.Module):
    def __init__(self, z_dim, out_channels):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self._block(z_dim, 1024),
            self._block(1024, 512),
            self._block(512, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
 
    def forward(self, x):
        return self.gen(x)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

if __name__ == "__main__":
    N, channels, H, W, z_dim = 32, 3, 128, 128, 100
    
    disc = Discriminator(channels)
    summary(disc,(channels,H,W),device="cpu")

    gen = Generator(z_dim, channels)
    # visualize_images(gen(torch.randn(32,z_dim,1,2)).detach())
    summary(gen,(z_dim,1,1),device="cpu")
