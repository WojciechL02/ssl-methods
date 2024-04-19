import torch.nn as nn
from .appr import Approach
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, 2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(256, 128, 3, 1)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(128, 64, 3, 1)
        self.act3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 32, 3, 1)
        self.act4 = nn.LeakyReLU()

    def forward(self, x):
        act1 = self.act1(self.conv1(x))
        act2 = self.act2(self.conv2(act1))
        act3 = self.act3(self.conv3(act2))
        act4 = self.act4(self.conv4(act3))
        return act1, act2, act3, act4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convt1 = nn.ConvTranspose2d(32, 64, 3, 1)
        self.act1 = nn.LeakyReLU()
        self.convt2 = nn.ConvTranspose2d(64, 128, 3, 1)
        self.act2 = nn.LeakyReLU()
        self.convt3 = nn.ConvTranspose2d(128, 256, 3, 1)
        self.act3 = nn.LeakyReLU()
        self.convt4 = nn.ConvTranspose2d(256, 3, 3, 2, output_padding=1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        act1 = self.act1(self.convt1(x))
        act2 = self.act2(self.convt2(act1))
        act3 = self.act3(self.convt3(act2))
        act4 = self.act4(self.convt4(act3))
        return act1, act2, act3, act4


class SAENet(nn.Module):
    def __init__(self):
        super(SAENet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        e_a1, e_a2, e_a3, e_a4 = self.encoder(x)
        d_a1, d_a2, d_a3, d_a4 = self.decoder(e_a4)
        return [e_a1, e_a2, e_a3, e_a4, d_a1, d_a2, d_a3, d_a4]


class SparsityMSE(nn.Module):
    def __init__(self, reduction: str, sparsity_penalty: float):
        super(SparsityMSE, self).__init__()
        self.sparsity_penalty = sparsity_penalty
        self._mse = nn.MSELoss(reduction=reduction)

    def forward(self, reconstruction, target, activations):
        mse = self._mse(reconstruction, target)
        l1 = self.sparsity_penalty * sum(torch.norm(actv, 1) for actv in activations)
        return mse + l1


class SparseAE(Approach):
    def __init__(self, device, nepochs, lr, logger, sparsity_penalty):
        super().__init__(device, SAENet(), nepochs, lr, logger)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = SparsityMSE("sum", sparsity_penalty)

    def _forward(self, data):
        target, _ = data[0].to(self.device), data[1].to(self.device)

        activations = self.model(target)
        reconstruction = activations[-1]
        loss = self.criterion(reconstruction, target, activations)
        return loss
