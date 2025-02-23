# -) ASSESSING DNA chromatin accessibility with ChromBPNet
# -) MPRA Predictor
# -) Metrics
# -) Correlation di motif predetti e reali
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numba
from scipy.spatial.distance import cdist
from numba.typed import Dict, List
from numba import types
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
    
    ref: DiscDiff, it searches for all n-grams with n \\in [10, 11, 12]

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

def prepare_lcs(ref, gen):
    ref_indices = np.array([prepare_single_seq_lcs(seq) for seq in ref])
    gen_indices = np.array([prepare_single_seq_lcs(seq) for seq in gen])
    print("ref_indices: ", ref_indices.shape)
    print("gen_indices: ", gen_indices.shape)
    return ref_indices, gen_indices

def prepare_single_seq_lcs(one_hot_seq):
    """
    This function prepares the input to the longest_alignment function.
    """
    # Convert one-hot encoded sequences to character sequences
    # The numpy array are N x 256 x 5
    # The third dimension is the one-hot encoding of the sequence in order "ACGTN"
    # We need to convert it to a sequence of characters
    return np.argmax(one_hot_seq, axis=-1)
    # Convert to list of strings

def filter_only_promising_pairs(ref, gen, th: float = 0.3):
    """
    Compute hamming distance"
    """
    dist_matrix = cdist(ref, gen, metric="hamming")
    return np.argwhere(dist_matrix < th)

@numba.njit(parallel=True)
def numba_compute_hamming_distance(ref, gen):
    """
    Compute the Hamming distance between two sets of one-hot encoded sequences.
    
    ref: (N, 256, 5) NumPy array
    gen: (M, 256, 5) NumPy array
    
    Returns:
        dist_matrix: (N, M) NumPy array with Hamming distances.
    """
    N, _ = ref.shape
    M, _ = gen.shape
    dist_matrix = np.zeros((N, M), dtype=np.float32)

    for i in numba.prange(N):
        for j in range(M):
            dist_matrix[i, j] = np.sum(ref[i] != gen[j]) / ref.shape[1]  # Hamming distance

    return dist_matrix

def compute_lcs(ref,gen):
    ref_indices, gen_indices = prepare_lcs(ref, gen)
    filtered_pairs = filter_only_promising_pairs(ref_indices, gen_indices)
    lcs_results = {}
    max_seq_length = 500
    for i, j in filtered_pairs:
        lcs_results[(i, j)] = longest_alignment(ref_indices[i], gen_indices[j], max_seq_length)
    return lcs_results


 #Define the types for the numba functions
key_type = numba.types.UniTuple(numba.types.int32, 2)
value_type = numba.types.ListType(types.UniTuple(types.int32,2))
@numba.njit(cache=True, parallel=True)
def numba_compute_lcs(ref_indices, gen_indices):
    dist_matrix = numba_compute_hamming_distance(ref_indices, gen_indices)

    # Filter pairs based on a distance threshold
    threshold = 0.3
    filtered_pairs = [(i, j) for i in range(dist_matrix.shape[0]) 
                              for j in range(dist_matrix.shape[1]) if dist_matrix[i, j] < threshold]

    # Define the type of results explicitly
    results = List()  # Use Numba-typed List
    for _ in range(len(filtered_pairs)):
        # Initialize each element as a tuple of the correct type
        results.append(((0, 0), List.empty_list(key_type)))

    # Parallel loop to compute LCS
    N = len(filtered_pairs)
    for idx in numba.prange(N):
        i, j = filtered_pairs[idx]
        lcs_pairs = numba_longest_alignment(ref_indices[i], gen_indices[j])

        typed_lcs_pairs = List.empty_list(key_type)
        for pair in lcs_pairs:
            typed_lcs_pairs.append(pair)
        
        key = (np.int32(i), np.int32(j))
        results[idx] = (key, typed_lcs_pairs)  # This is now type-consistent

    # Convert results to a dictionary
    lcs_results = Dict.empty(
        key_type=key_type,
        value_type=value_type,
    )
    for (i, j), lcs in results:
        lcs_results[(i, j)] = lcs

    return lcs_results
   
def longest_alignment(A, B):
    """
    This function computes the length of the longest alignment between two sets of DNA sequences.
    """
    import bisect
    # reconvert array to sequence of characters
    m, n = len(A), len(B)

    # Step 1: build linked lists
    matchlist = [[] for k in range(m + 1)]
    # Note line numbers in reverse order
    aa = sorted(zip(A, range(1, m+1)), key=lambda t: (t[0], -t[1]))
    bb = sorted(zip(B, range(1, n+1)), key=lambda t: (t[0], -t[1]))
    ai = bi = 0
    while ai < m and bi < n:
        av, bv = aa[ai][0], bb[bi][0]
        if av < bv:
            ai += 1
        elif av > bv:
            bi += 1
        else:
            k = aa[ai][1]
            while bi < n and bb[bi][0] == bv:
                matchlist[k] += [bb[bi][1]]
                bi += 1
            ai += 1
            while ai < m and aa[ai][0] == av:
                matchlist[aa[ai][1]] = matchlist[k]
                ai += 1

    # Step 2: initialize the THRESH array
    thresh = [n+1] * (m + 1)
    thresh[0] = 0

    # Step 3: compute successive THRESH values
    link = [None] * (m+1)
    for i in range(1, m+1):
        for j in matchlist[i]:
            #find k such that thresh[k-1] < j <= thresh[k]
            k = bisect.bisect_left(thresh, j)
            #assert thresh[k-1] < j <= thresh[k]
            if j < thresh[k]:
                thresh[k] = j
                link[k] = (i, j, link[k-1])
                #print(f'dmatch({i}, {j})')
                #assert A[i-1] == B[j-1]

    # Step 4: recover longest common subsequence pairs in reverse order
    k = 0
    while k < m and thresh[k+1] != n + 1:
        k += 1
    p = link[k]
    # v will hold (i,j) pairs
    v = []
    while p != None:
        v.append(p[:2])
        p = p[2]
    v.reverse()

    #print(f'lcslen: {len(v)=}')
    return v

@numba.njit(cache=True)
def numba_longest_alignment(A, B):
    m, n = len(A), len(B)

    # Define the type of matchlist explicitly
    matchlist = List()
    for _ in range(m + 1):
        matchlist.append(List.empty_list(types.int32))  # Inner lists are of type List(int32)
    # Prepare NumPy arrays for sorting
    aa = np.zeros((m, 2), dtype=np.int32)
    bb = np.zeros((n, 2), dtype=np.int32)
    
    for i in range(m):
        aa[i, 0] = A[i]
        aa[i, 1] = i + 1
    for i in range(n):
        bb[i, 0] = B[i]
        bb[i, 1] = i + 1

    aa_sorted = aa[np.argsort(aa[:, 0])]
    bb_sorted = bb[np.argsort(bb[:, 0])]

    # Build matchlist
    ai = bi = 0
    while ai < m and bi < n:
        av, bv = aa_sorted[ai, 0], bb_sorted[bi, 0]
        if av < bv:
            ai += 1
        elif av > bv:
            bi += 1
        else:
            k = aa_sorted[ai, 1]
            matches = List.empty_list(types.int32)
            while bi < n and bb_sorted[bi, 0] == bv:
                matches.append(bb_sorted[bi, 1])
                bi += 1
            matchlist[k] = matches
            ai += 1

    # Initialize THRESH array
    thresh = np.full(m + 1, n + 1, dtype=np.int32)
    thresh[0] = 0

    # Initialize link array
    link = np.full((m + 1, 2), -1, dtype=np.int32)

    # Compute successive THRESH values
    for i in range(1, m + 1):
        if matchlist[i]:
            for j in matchlist[i]:
                k = 0
                while k < m and thresh[k] < j:
                    k += 1
                if j < thresh[k]:
                    thresh[k] = j
                    link[k] = [i, j]

    # Recover longest common subsequence pairs
    k = 0
    while k < m and thresh[k + 1] != n + 1:
        k += 1

    # Backtrace
    v = []
    p = k
    while p >= 0:
        if link[p, 0] != -1:
            v.append((link[p, 0], link[p, 1]))
        p -= 1

    return v[::-1]  # Reverse order

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

def motif_correlation(one_hot1, one_hot2, motif:str="TATAWAW", onehot: str = "ACGTN", figname:str = "motif_correlation.png"):
    distr1 = find_subsequence(one_hot1, motif, onehot).mean(axis=0) if isinstance(one_hot1, np.ndarray) else find_subsequence(one_hot1, motif, onehot).float().mean(dim=0)
    distr2 = find_subsequence(one_hot2, motif, onehot).mean(axis=0) if isinstance(one_hot2, np.ndarray) else find_subsequence(one_hot2, motif, onehot).float().mean(dim=0)

    # Plot the motif frequency along the sequence
    # TODO: fix style of the plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0] = plot_motif_frequence_position(distr1, fig, ax[0])
    ax[1] = plot_motif_frequence_position(distr2, fig, ax[1]) 
    fig.tight_layout()
    fig.savefig(figname)

    # TODO: do we want to use the average position across sequences of each dataset or the frequency to evaluate correlation?
    #average_distr1 = distr1 / one_hot1.shape[0]
    #average_distr2 = distr2 / one_hot2.shape[0]
    if isinstance(one_hot1, np.ndarray) and isinstance(one_hot2, np.ndarray):
        return np.corrcoef(distr1, distr2)[0,1]
    elif isinstance(one_hot1, torch.Tensor) and isinstance(one_hot2, torch.Tensor):
        return torch.corrcoef(torch.stack((distr1, distr2)))[0,1]
    else:
        raise TypeError("Both inputs must be of the same type, either numpy arrays or torch tensors")


if __name__ == "__main__":
    # Generate a tensor of 100x5 one hot encoded sequences
    import os
    from pathlib import Path
    current_dir = Path(os.getcwd())
    data_dir = current_dir / "data"
    ref = np.load(data_dir / "dataset_compressed.npz")
    gen = np.load(data_dir / "generated.npz")

    print(ref.keys, ref["data"].shape, type(ref))
    print(gen.keys, gen["data"].shape, type(gen))
    
    # WORK WITH A SUBSET
    N_list = [1e3, 1e4, 1e5, 1e6]
    for N in N_list:
        N = int(N)
        print(f"Working with a subset of {N} sequences")
    # randomly sampled without replacement
        if N > ref["data"].shape[0] or N > gen["data"].shape[0]:
            N = min(ref["data"].shape[0], gen["data"].shape[0])
        ref_indices = np.random.choice(ref["data"].shape[0], N, replace=False)
        gen_indices = np.random.choice(gen["data"].shape[0], N, replace=False)
        ref_data = ref["data"][ref_indices]
        gen_data = gen["data"][gen_indices]
         #convert tensors to numpy array
        #print("Computing motif correlation...")
        #tic = time.time()
        #corr = motif_correlation(ref_data, gen_data, figname=f"motif_correlation_{N}.png")
        #toc = time.time()
        #print(f"Motif correlation computed in {toc - tic:.2f}s as value {corr}")
        #print("Computing diversity...")
        #tic = time.time()
        #gen = gen.numpy()
        #ref = ref.numpy()
        #div_perc_delta = delta_diversity(ref, gen)
        #toc = time.time()
        #print(f"Diversity of generated sequences and ref sequences in {toc-tic:.2f}s with delta {div_perc_delta:.3f}%")
        #toc = time.time()
        print("Computing longest alignment...")
        tic = time.time()
        ref_indices = np.array([np.argmax(seq,axis=-1) for seq in ref_data])
        gen_indices = np.array([np.argmax(seq,axis=-1) for seq in gen_data])

        lcs_results = numba_compute_lcs(ref_indices, gen_indices)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist([len(lcs_results[k]) for k in lcs_results], bins=50)
        ax.set_title("Length of longest alignment")
        ax.set_xlabel("Length")
        ax.set_ylabel("Frequency")
        fig.savefig(f"longest_alignment_{N}.png")
        toc = time.time()
        print(f"Longest alignment computed in {toc - tic:.2f}s")
