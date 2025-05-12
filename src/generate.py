import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
from models import DiffusionTransformer
from datamodules import DiscDiffDataModule, DHSDataModule


def generate(model, dataloader, device):
    xs = []
    ys = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)

            x0 = torch.randn_like(x)
            x1 = model.push(x0, y, n_steps=8)

            xs.append(x1.cpu().numpy())
            ys.append(y.cpu().numpy().astype(bool))

    y = np.concatenate(ys, axis=0)
    x = np.concatenate(xs, axis=0)
    x = x == x.max(axis=-1, keepdims=True)
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["promoters", "dhs"])
    parser.add_argument("--full", type=bool, default=True)
    args = parser.parse_args()

    # Load model checkpoint 
    ckpt_path = os.path.join(os.getcwd(), "data", f"{args.data}", "model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DiffusionTransformer.load_from_checkpoint(ckpt_path, map_location=device).eval()
    
    # Load datamodule
    datamodule = DiscDiffDataModule()
    dataloader = datamodule.test_dataloader()
    if args.full:
        dataloader = [*datamodule.train_dataloader(), *datamodule.test_dataloader()]

    # Generate and save sequences
    data, labels = generate(model, dataloader, device)
    save_path = os.path.join(os.getcwd(), "data", f"{args.data}", "generated.npz")
    np.savez_compressed(save_path, data=data, labels=labels)
