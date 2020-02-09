from torch import nn

from .layers import Reshape


class ConvDecoder(nn.Module):
    def __init__(self, z_dim=16):
        super(ConvDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 32),
            nn.ReLU(),
            Reshape((-1, 32, 7, 7)),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class FcDecoder(nn.Module):
    def __init__(self, z_dim=16):
        super(FcDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 784),
            nn.Sigmoid(),
            Reshape((-1, 1, 28, 28))
        )

    def forward(self, z):
        return self.decoder(z)
