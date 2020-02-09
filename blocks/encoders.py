from torch import nn
from torch.nn.modules.flatten import Flatten


class ConvEncoder(nn.Module):
    def __init__(self, h_dim=100, z_dim=30):
        super(ConvEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(2304, h_dim)
        )

        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar


class FcEncoder(nn.Module):
    def __init__(self, z_dim=30):
        super(FcEncoder, self).__init__()

        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU()
        )

        self.fc21 = nn.Linear(2000, z_dim)
        self.fc22 = nn.Linear(2000, z_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar
