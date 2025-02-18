# -) ASSESSING DNA chromatin accessibility with ChromBPNet
# -) MPRA Predictor
# -) Metrics
# -) Correlation di motif predetti e reali
import torch
import numpy as np
import time


######## DIVERSITY ########
def count_ngrams(one_hot: np.ndarray, n: int):
    """Efficiently counts unique n-grams using vectorized operations."""
    Nobs, seq_len, vocab_size = one_hot.shape
    assert vocab_size == 5, "Unexpected vocabulary size, expected 5 for DNA bases."

    # Precompute base-5 weights for encoding n-grams into unique integers
    base_values = (5 ** np.arange(vocab_size))[::-1]  # Example: [5^4, 5^3, 5^2, 5^1, 5^0]

    # Generate all n-grams using NumPy's sliding window trick
    shape = (Nobs, seq_len - n + 1, n, vocab_size)  # Reshape to extract all n-grams
    strides = (*one_hot.strides[:2], *one_hot.strides[1:])  # Sliding window strides
    ngrams = np.lib.stride_tricks.as_strided(one_hot, shape=shape, strides=strides)

    # Encode each n-gram to an integer for fast uniqueness checking
    encoded_ngrams = np.dot(ngrams, base_values).sum(axis=-1)  # Shape: (Nobs, seq_len - n + 1)

    # Flatten across observations to compute unique n-grams
    unique_ngrams = np.unique(encoded_ngrams)  # Fast unique count
    total_ngrams = encoded_ngrams.size

    return unique_ngrams, total_ngrams

def diversity(one_hot: torch.Tensor | np.ndarray, n_min: int = 10, n_max: int = 12):
    """
    This funciton should compute the diversity between two seuqences, defined as:
    product of ratios between the unique amount of n-grams found in a set of generated DNA sequences D
    and the number of n-grams found.
    
    ref: DiscDiff, it searches for all n-grams with n \in [10, 11, 12]

    This metric assesses the variety within the generated DNA sequences
    """
    div = 1.0
    for n in range(n_min, n_max + 1):
        unique_ngrams, total_ngrams = count_ngrams(one_hot, n)
        if total_ngrams == 0:  # Prevent division by zero
            continue
        #print(f"unique n-grams at iteration {n}: {len(set(unique_ngrams))} out of {total_ngrams}")
        div *= (len(set(unique_ngrams)) / total_ngrams)
        #print(f"div at iteration {n}: {div}")
    return div

def delta_diversity(one_hot1, one_hot2, n_min: int = 10, n_max: int = 12):
    """
    This function computes the difference in diversity between two sets of DNA sequences.
    """
    div1 = diversity(one_hot1, n_min, n_max)
    div2 = diversity(one_hot2, n_min, n_max)
    return (div2-div1)/div1*100

######## FRECHET INCEPTION DISTANCE ########
def s_fid_inception_distance():
    """
    This function computes the adapted score from the FID used in image generation.
    S-FID metric measures the distance between the distributions of generated and natural DNA
    sequences in teh latent space. It uses the encoder of a pre-trained genomic neural network.
    """
    #TODO: implement the function
    pass

######### LENGTH OF LONGEST ALIGNMENT ########
def longest_alignment(ref: np.ndarray | torch.Tensor, gen: np.ndarray | torch.Tensor):
    """
    This function computes the length of the longest alignment between two sets of DNA sequences.
    """
    pass

######## MOTIF CORRELATION ########
def find_subsequence(one_hot, subsequence: str, char_list: str = "ACGNT"):
    if isinstance(one_hot, np.ndarray):
        masks = {k: one_hot[..., i].astype(bool) for i, k in enumerate(char_list)}
        masks["W"] = masks["A"] | masks["T"]
        matches = np.ones_like(one_hot[..., 0], dtype=bool)
        for i, char in enumerate(subsequence):
            matches &= np.roll(masks[char], -i, axis=-1)
    elif isinstance(one_hot, torch.Tensor):
        masks = {k: one_hot[..., i].bool() for i, k in enumerate(char_list)}
        masks["W"] = masks["A"] | masks["T"]
        matches = torch.ones_like(one_hot[..., 0], dtype=torch.bool)
        for i, char in enumerate(subsequence):
            matches &= torch.roll(masks[char], shifts=-i, dims=-1)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")
    return matches

def plot_motif_frequence_position(distr, fig, ax):
    ax.plot(range(0,distr.shape[0]),distr)
    ax.set_title("Motif frequency along the sequence")
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    return ax

def motif_correlation(one_hot1, one_hot2, motif:str="TATAWAW", onehot: str = "ACGTN"):
    distr1 = find_subsequence(one_hot1, motif, onehot).sum(axis=0) if isinstance(one_hot1, np.ndarray) else find_subsequence(one_hot1, motif, onehot).float().sum(dim=0)
    distr2 = find_subsequence(one_hot2, motif, onehot).sum(xis=0) if isinstance(one_hot2, np.ndarray) else find_subsequence(one_hot2, motif, onehot).float().sum(dim=0)

    # Plot the motif frequency along the sequence
    # TODO: fix style of the plot
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0] = plot_motif_frequence_position(distr1, fig, ax[0])
    ax[1] = plot_motif_frequence_position(distr2, fig, ax[1]) 
    fig.tight_layout()
    fig.savefig("distr.png")

    # TODO: do we want to use the average position across sequences of each dataset or the frequency to evaluate correlation?
    average_distr1 = distr1 / one_hot1.shape[0]
    average_distr2 = distr2 / one_hot2.shape[0]
    if isinstance(one_hot1, np.ndarray) and isinstance(one_hot2, np.ndarray):
        return np.corrcoef(distr1, distr2)[0,1]
    elif isinstance(one_hot1, torch.Tensor) and isinstance(one_hot2, torch.Tensor):
        return torch.corrcoef(torch.stack((distr1, distr2)))[0,1]
    else:
        raise TypeError("Both inputs must be of the same type, either numpy arrays or torch tensors")

if __name__ == "__main__":
    # Generate a tensor of 100x5 one hot encoded sequences
    ref = torch.randint(0, 2, (1000, 1000, 5))
    gen = torch.randint(0, 2, (1000, 1000, 5))
    # convert tensors to numpy array
    print("Computing motif correlation...")
    tic = time.time()
    corr = motif_correlation(ref, gen)
    toc = time.time()
    print(f"Motif correlation computed in {toc - tic:.2f}s as value {corr}")
    print("Computing diversity...")
    tic = time.time()
    gen = gen.numpy()
    ref = ref.numpy()
    div_perc_delta = delta_diversity(ref, gen)
    toc = time.time()
    print(f"Diversity of generated sequences and ref sequences in {toc-tic:.2f}s with delta {div_perc_delta:.3f}%")
    toc = time.time()
