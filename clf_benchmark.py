import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_datasets
from approaches.dae import DAENet
from approaches.mae import MAENet
from approaches.sae import SAENet
from approaches.simclr import SimCLRNet


class Classifier(nn.Module):
    def __init__(self, encoder, latent_size):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(latent_size, 10)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_clf(approach, device):
    if approach == "dae":
        dae = DAENet()
        dae.load_state_dict(torch.load("checkpoints/mae_17:55.pt", map_location=device))
        clf = Classifier(dae.encoder, 2592)
    elif approach == "mae":
        mae = MAENet()
        mae.load_state_dict(torch.load("checkpoints/mae_17:55.pt", map_location=device))
        clf = Classifier(mae.encoder, 100)
    elif approach == "sae":
        sae = SAENet()
        sae.load_state_dict(torch.load("checkpoints/sae_17:55.pt", map_location=device))
        clf = Classifier(sae.encoder, 100)
    elif approach == "simclr":
        simclr = SimCLRNet()
        simclr.load_state_dict(torch.load("checkpoints/simclr_17:55.pt", map_location=device))
        clf = Classifier(simclr.encoder, 100)
    else:
        raise ValueError(f"No such approach: {approach}")
    return clf


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset, testset = get_datasets("cifar10")
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=True)

    approaches = ["dae", "mae", "sae", "simclr"]

    for approach in approaches:
        model = get_clf(approach, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # TRAINING
        for e in range(10):
            model.train()
            for data, targets in trainloader:
                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

        # EVALUATION
        model.eval()
        with torch.no_grad():
            total_loss = 0.
            hits = 0
            num_samples = 0
            for data, targets in testloader:
                logits = model(data)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                _, y_pred = torch.max(logits.data, 1)
                hits += (y_pred == targets).sum().item()
                num_samples += len(targets)

            final_loss = total_loss / len(testloader)
            acc = hits / num_samples
            print(f"Approach: {approach} | Loss: {final_loss:.4f} | Accuracy: {100 * acc:.1f}")


if __name__ == "__main__":
    main()
