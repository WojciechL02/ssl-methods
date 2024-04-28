from .appr import Approach
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self.mask_correlated_samples()

    def mask_correlated_samples(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        pairwise_sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(pairwise_sim, self.batch_size)
        sim_j_i = torch.diag(pairwise_sim, -self.batch_size)

        positive = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative = pairwise_sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(z_i.device).long()
        logits = torch.cat((positive, negative), dim=1)
        loss = self.criterion(logits, labels) / N
        return loss


class SimCLRNet(nn.Module):
    def __init__(self):
        super(SimCLRNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


class SimCLR(Approach):
    def __init__(self, device, nepochs, lr, logger, batch_size, temperature):
        super().__init__(device, SimCLRNet(), nepochs, lr, logger)
        self._appr_name = "simclr"
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = NT_Xent(batch_size, temperature)

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
        ])

    def _forward(self, data):
        data, _ = data[0].to(self.device), data[1].to(self.device)
        x_i = self.transforms(data)
        x_j = self.transforms(data)

        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss
