import torch.nn as nn
from .appr import Approach
import torch


class DAENet(nn.Module):
    def __init__(self):
        super(DAENet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 128, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 3, 3, 2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.encoder(x)
        return features


class DAE(Approach):
    def __init__(self, device, nepochs, lr, logger, noise_std=0.3, noise_mean=0.):
        super().__init__(device, DAENet(), nepochs, lr, logger)
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def _forward(self, data):
        data, _ = data[0].to(self.device), data[1].to(self.device)
        noise = torch.randn(data.size()) * self.noise_std + self.noise_mean
        target = data + noise

        features = self.model(data)
        reconstruction = self.model.decoder(features)
        loss = self.criterion(reconstruction, target)
        return loss
