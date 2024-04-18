import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from approaches.dae import DAE
from approaches.simclr import SimCLR


def get_datasets(dataset: str):
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = CIFAR10(root="../datasets/cifar10", train=True, transform=transform, download=False)
        testset = CIFAR10(root="../datasets/cifar10", train=False, transform=transform, download=False)
        return trainset, testset

    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = MNIST(root="../datasets/mnist", train=True, transform=transform, download=False)
        testset = MNIST(root="../datasets/mnist", train=False, transform=transform, download=False)
        return trainset, testset

    return None


def get_approach(approach, *args, **kwargs):
    if approach == "dae":
        return DAE(*args, **kwargs)
    elif approach == "simclr":
        return SimCLR(*args, **kwargs)
    return None


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
