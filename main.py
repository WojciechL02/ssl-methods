import argparse
import yaml
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_datasets, get_approach


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.config_path}", "r") as config_file:
        exp_args = yaml.load(config_file, Loader=yaml.FullLoader)

    trainset, testset = get_datasets(exp_args["dataset"])
    trainloader = DataLoader(trainset, batch_size=exp_args["batch_size"], shuffle=True, num_workers=3, drop_last=True)
    testloader = DataLoader(testset, batch_size=exp_args["batch_size"], shuffle=False, num_workers=3, drop_last=True)

    # ================= PREPARE APPROACH =================
    curtime = datetime.now().strftime("%b%d_%H-%M-%S")
    exp_logger = SummaryWriter(log_dir=f"runs/{exp_args['approach']}/{curtime}")
    appr_kwargs = dict(nepochs=exp_args["nepochs"], lr=exp_args["lr"], logger=exp_logger, patch_size=exp_args["patch_size"], masking_ratio=exp_args["masking_ratio"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    appr = get_approach(exp_args["approach"], device, **appr_kwargs)

    # ================= TRAIN =================
    appr.train(trainloader, testloader)

    exp_logger.flush()
    exp_logger.close()

    # ================= TEST =================
    x = testset[0][0]

    appr.model.eval()
    y = appr.model.encoder(x)
    print(y.shape)


if __name__ == "__main__":
    main()
