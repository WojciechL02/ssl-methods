import torch.nn as nn
import torch
from .appr import Approach


class MAENet(nn.Module):
    def __init__(self):
        super(MAENet, self).__init__()
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
        return reconstruction


class MAE(Approach):
    def __init__(self, device, nepochs, lr, logger, patch_size, masking_ratio):
        super(MAE, self).__init__(device, MAENet(), nepochs, lr, logger)
        self._appr_name = "mae"
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.MSELoss(reduction="sum")

    def apply_masking(self, x):
        B, C, H, W = x.size()
        n_patches_h = (H // self.patch_size)
        n_patches = n_patches_h ** 2
        n_masked = int(self.masking_ratio * n_patches)
        masked_ids = torch.randperm(n=n_patches)[:n_masked]

        mask = torch.ones((B, 1, H, W)).to(x.device)
        for idx in masked_ids:
            idx = idx.item()
            patch_row = idx // n_patches_h
            patch_col = idx % n_patches_h
            start_row = patch_row * self.patch_size
            start_col = patch_col * self.patch_size
            mask[:, :, start_row:start_row + self.patch_size, start_col:start_col + self.patch_size] = 0.
        masked_x = x * mask
        return masked_x

    def _forward(self, data):
        target, _ = data[0].to(self.device), data[1].to(self.device)
        data = self.apply_masking(target)

        reconstruction = self.model(data)
        loss = self.criterion(reconstruction, target)
        return loss
