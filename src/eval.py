import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def find_subsequence(one_hot: np.ndarray, subsequence: str, char_list: str = "ACGTN"):
    masks = {k: one_hot[..., i].astype(bool) for i, k in enumerate(char_list)}
    masks["W"] = masks["A"] | masks["T"]
    matches = np.ones_like(one_hot[..., 0], dtype=bool)
    for i, char in enumerate(subsequence):
        matches &= np.roll(masks[char], -i, axis=-1)
    return matches


def smooth(x, w=0.01):
    w = int(w * len(x))
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_motif(motif: str, real: np.ndarray, generated: np.ndarray, random: bool = True):
    plt.title(f"{motif} motif")
    plt.semilogy(smooth(find_subsequence(real, motif).mean(axis=0)), label="real")
    plt.semilogy(smooth(find_subsequence(generated, motif).mean(axis=0)), label="generated")
    if random:
        random = np.ones_like(real[0]) * (1/4) ** len(motif)
        plt.hlines(random, 0, len(random), color="k", label="random")
    plt.legend()
    plt.grid()


# def motif_correlation(one_hot1, one_hot2, motif:str="TATAWAW", onehot: str = "ACGTN"):
#     distr1 = find_subsequence(one_hot1, motif, onehot).mean(axis=0)
#     distr2 = find_subsequence(one_hot2, motif, onehot).mean(axis=0)
#     return corr(distr1, distr2) # questa riga non va


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["promoters", "dhs"])
    args = parser.parse_args()

    # Load real sequences data
    real_file = np.load(os.path.join(os.getcwd(), "data", f"{args.data}", "dataset.npz"))
    real_data, real_labels = real_file["data"], real_file["labels"]
    print("Real data: ", real_data.shape, real_data.dtype)
    print("Real labels:", real_labels.shape, real_labels.dtype)
    assert (real_labels.astype(int).sum(-1) <= 1).all(), "Labels should be one-hot encoded"

    # Load generated sequences data
    generated_file = np.load(os.path.join(os.getcwd(), "data", f"{args.data}", "generated.npz"))
    generated_data, generated_labels = generated_file["data"], generated_file["labels"]
    print("Generated data:", generated_data.shape, generated_data.dtype)
    print("Generated labels:", generated_labels.shape, generated_labels.dtype)
    assert (generated_labels.astype(int).sum(-1) <= 1).all(), "Labels should be one-hot encoded"

    for motif in ["TATAWAW", "GGGCGG"]:
        # Plot motifs distribution global
        plt.figure(figsize=(10, 5))
        plot_motif(motif, real_data, generated_data)
        plt.savefig(os.path.join(os.getcwd(), "figures", f"{args.data}", f"motif={motif}_all.pdf"), bbox_inches="tight")
        plt.close()

        # Plot motifs distribution for each label
        for label in tqdm(range(real_labels.shape[1])):
            # Skip if label is not present in real data
            if not real_labels[:, label].any():
                continue
            plt.figure(figsize=(10, 5))
            real_data_for_label = real_data[real_labels[:, label]]
            generated_data_for_label = generated_data[generated_labels[:, label]]
            plot_motif(motif, real_data_for_label, generated_data_for_label)
            plt.savefig(os.path.join(os.getcwd(), "figures", f"{args.data}", f"motif={motif}_label={label}.pdf"), bbox_inches="tight")
            plt.close()

    