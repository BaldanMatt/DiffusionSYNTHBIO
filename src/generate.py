import os
from tqdm import tqdm
import torch
import numpy as np
from models import DiffusionTransformer
from datamodules import DiscDiffDataModule, DHSDataModule


def generate(model, datamodule, device):
    xs = []
    ys = []
    with torch.no_grad():
        for x, y in tqdm(datamodule.test_dataloader()):
            x, y = x.to(device), y.to(device)

            x0 = torch.randn_like(x)
            x1 = model.push(x0, y, n_steps=16)

            xs.append(x1.cpu().numpy())
            ys.append(y.cpu().numpy().astype(bool))

    y = np.concatenate(ys, axis=0)
    x = np.concatenate(xs, axis=0)
    x = x == x.max(axis=-1, keepdims=True)
    return x, y


if __name__ == "__main__":
    ckpt_path = os.path.join(os.getcwd(), "ckpt", "discdiff_L.ckpt")
    save_path = os.path.join(os.getcwd(), "generated", "discdiff_L.npz")
    datamodule = DiscDiffDataModule()

    device = torch.device("cuda:2")
    model = DiffusionTransformer.load_from_checkpoint(ckpt_path, map_location=device).eval()
    data, labels = generate(model, datamodule, device)
    np.savez_compressed(save_path, data=data, labels=labels)
