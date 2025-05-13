import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import DiffusionTransformer
from datamodules import EPDGenDNA_2048, EPDGenDNA_256


def generate(model: DiffusionTransformer, dataloader):
    xs = []
    ys = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Generating sequences"):
            x, y = x.to("cuda"), y.to("cuda")
            x0 = torch.randn_like(x)
            x1 = model.push(x0, y, n_steps=8)
            xs.append(x1.cpu().numpy())
            ys.append(y.cpu().numpy())
    y = np.concatenate(ys, axis=0)
    x = np.concatenate(xs, axis=0)
    x = x == x.max(-1, keepdims=True)
    y = y.astype(bool)
    return x, y


def find_subsequence(one_hot: np.ndarray, subsequence: str, char_list: str):
    subsequence = list(np.array(list(subsequence), dtype='|S1'))
    masks = {k: one_hot[..., i].astype(bool) for i, k in enumerate(char_list)}
    masks[b"W"] = masks[b"A"] | masks[b"T"]
    matches = np.ones_like(one_hot[..., 0], dtype=bool)
    for i, char in enumerate(subsequence):
        matches &= np.roll(masks[char], -i, axis=-1)
    return matches


def plot_motif_distr(motif: str, real: np.ndarray, generated: np.ndarray, char_list: str, random: bool = True,):
    def smooth(x, w=0.01):
        w = int(w * len(x))
        return np.convolve(x, np.ones(w), 'valid') / w

    plt.title(f"{motif} motif")
    plt.semilogy(smooth(find_subsequence(real, motif, char_list).mean(axis=0)), label="real")
    plt.semilogy(smooth(find_subsequence(generated, motif, char_list).mean(axis=0)), label="generated")
    if random:
        random = np.ones_like(real[0]) * (1/4) ** len(motif)
        plt.hlines(random, 0, len(random), color="k", label="random")
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Dataset name")
    args = parser.parse_args()
    save_dir = os.path.join(os.getcwd(), f"{args.data}")

    # Load datamodule
    datamodule = getattr(__import__("datamodules"), args.data)()
    real_x, real_y = datamodule.x_test.numpy(), datamodule.y_test.numpy().astype(bool)

    save_dir = os.path.join(os.getcwd(), f"{args.data}")
    if not os.path.exists(os.path.join(save_dir, "generated.npz")):
        # Load datamoduledatamodule and model checkpoint 
        datamodule = getattr(__import__("datamodules"), args.data)()
        ckpt_path = os.path.join(save_dir, "model.ckpt")
        model = DiffusionTransformer.load_from_checkpoint(ckpt_path, map_location=torch.device("cuda")).eval()

        # Generate and save sequences
        generated_x, generated_y = generate(model, datamodule.test_dataloader())
        np.savez_compressed(os.path.join(save_dir, "generated.npz"), x=generated_x, y=generated_y)
    else:
        # Load generated sequences data
        generated_file = np.load(os.path.join(save_dir, "generated.npz"))
        generated_x, generated_y = generated_file["x"], generated_file["y"]
    
    # Create save directory
    save_dir = os.path.join(os.path.join(save_dir, "figures"))
    os.makedirs(save_dir, exist_ok=True)
    
    for motif in ["TATAWAW", "GGGCGG"]:
        # Plot motifs distribution global
        plt.figure(figsize=(10, 5))
        plot_motif_distr(motif, real_x, generated_x, datamodule.x_descr, random=True)
        plt.savefig(os.path.join(save_dir, f"{motif}_all.pdf"), bbox_inches="tight")
        plt.close()

        # Plot motifs distribution for each label
        for idx, label in tqdm(enumerate(datamodule.y_descr)):
            plt.figure(figsize=(10, 5))
            plot_motif_distr(motif, real_x[real_y[:, idx]], generated_x[generated_y[:, idx]], datamodule.x_descr, random=True)
            plt.savefig(os.path.join(save_dir, f"{motif}_{label}.pdf"), bbox_inches="tight")
            plt.close()

    