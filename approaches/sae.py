import torch.nn as nn
from .appr import Approach
import torch


class SAENet(nn.Module):
    def __init__(self):
        super(SAENet, self).__init__()
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return features, reconstruction


class SparsityMSE(nn.Module):
    def __init__(self, reduction: str, sparsity_penalty: float):
        super(SparsityMSE, self).__init__()
        self.sparsity_penalty = sparsity_penalty
        self._mse = nn.MSELoss(reduction=reduction)

    def forward(self, reconstruction, target, activations):
        mse = self._mse(reconstruction, target)
        l1 = self.sparsity_penalty * torch.norm(activations, 1)
        return mse + l1


class SparseAE(Approach):
    def __init__(self, device, nepochs, lr, logger, sparsity_penalty):
        super().__init__(device, SAENet(), nepochs, lr, logger)
        self._appr_name = "sae"
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = SparsityMSE("sum", sparsity_penalty)

    def _forward(self, data):
        target, _ = data[0].to(self.device), data[1].to(self.device)

        features, reconstruction = self.model(target)
        loss = self.criterion(reconstruction, target, features)
        return loss
