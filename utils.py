import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import torch.nn.functional as F
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from approaches.dae import DAE
from approaches.mae import MAE
from approaches.simclr import SimCLR
from approaches.sae import SparseAE


def get_datasets(dataset: str):
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        trainset = CIFAR10(root="../datasets/cifar10", train=True, transform=transform, download=False)
        testset = CIFAR10(root="../datasets/cifar10", train=False, transform=transform, download=False)
        return trainset, testset

    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = MNIST(root="../datasets/mnist", train=True, transform=transform, download=False)
        testset = MNIST(root="../datasets/mnist", train=False, transform=transform, download=False)
        return trainset, testset
    else:
        raise ValueError(f"No such dataset: {dataset}")


def get_approach(approach, *args, **kwargs):
    if approach == "dae":
        return DAE(*args, **kwargs)
    elif approach == "simclr":
        return SimCLR(*args, **kwargs)
    elif approach == "mae":
        return MAE(*args, **kwargs)
    elif approach == "sae":
        return SparseAE(*args, **kwargs)

    return None


def get_approach_keys(approach_name):
    if approach_name == "dae":
        return ["nepochs", "lr", "logger", "noise_std", "noise_mean"]
    elif approach_name == "mae":
        return ["nepochs", "lr", "logger", "patch_size", "masking_ratio"]
    elif approach_name == "sae":
        return ["nepochs", "lr", "logger", "sparsity_penalty"]
    elif approach_name == "simclr":
        return ["nepochs", "lr", "logger", "batch_size", "temperature"]

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


def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    labels = []
    for imgs, label in data_loader:
        with torch.no_grad():
            z = model.encoder(imgs).flatten(1)
        img_list.append(imgs)
        embed_list.append(z)
        labels.append(label)
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(labels, dim=0))


def find_similar_images(query_img, query_z, key_embeds, K=8):
    # dist = torch.cdist(query_z[None,:], key_embeds[1], p=2)
    dist = torch.nn.functional.cosine_similarity(query_z[None, :], key_embeds[1])
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img.unsqueeze(0).cpu(), key_embeds[0][indices.cpu()[:K]]], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K+1, normalize=True)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12, 3))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def visualize_reconstructions(model, input_imgs, device):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)  # , range=(-1,1)
    grid = grid.permute(1, 2, 0)
    if len(input_imgs) == 4:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(15, 10))
    plt.title(f"Reconstructions")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def plot_latent_space(data, approach):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(
        x=0, y=1,
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.9
    ).set(title=f"{approach} latent space")
    plt.show()


def plot_latent_space_with_annotations(data, examples, examples_locations, approach):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(
        x=0, y=1,
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.1
    ).set(title=f"{approach} latent space")
    for location, example in zip(examples_locations, examples):
        x, y = location[0], location[1]
        label = int(location["label"])
        ab = AnnotationBbox(OffsetImage(np.swapaxes(np.swapaxes(example, 0, 1), 1, 2) / 2 + 0.5, zoom=1), (x, y),
                            frameon=True,
                            bboxprops=dict(facecolor=sns.color_palette("hls", 10)[label], boxstyle="round"))
        ax.add_artist(ab)
    plt.show()
