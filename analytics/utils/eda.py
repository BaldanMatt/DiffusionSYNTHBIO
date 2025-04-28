import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pathlib, os
def load_data(metadata_path, data_path, dim=1000):

    # Load metadata from CSV
    # Load data assuming that the first column is not the index
    metadata = pd.read_csv(metadata_path, index_col=0)
    print("Metadata loaded successfully.")
    print(metadata.info())
    print(metadata.describe())

    # Load data from .npz
    data = np.load(data_path)
    print("Data loaded successfully.")
    print(f"Keys: {list(data.keys())}")
    for key in data.keys():
        print(f"{key}: {data[key].shape}")

    num_labels = data["y"].shape[1]
    sampled_indices = []
    for i in range(num_labels):
        label_indices = np.where(data["y"][:, i] == 1)[0]
        # Sample the specified number of indices
        sampled = np.random.choice(label_indices, size=dim, replace=False)
        sampled_indices.extend(sampled)
    # Get the sampled data
    x = data["X"][sampled_indices,:,:]
    y = data["y"][sampled_indices,:]
    z = data["widths"][sampled_indices]
    data = {"X": x, "y": y, "widths": z}
    metadata = metadata.iloc[sampled_indices,:]

    print(f"Sampled {len(sampled_indices)} indices.")
    print(f"Sampled data shape: {data['X'].shape}")
    print(f"Sampled metadata shape: {metadata.shape}")
    print(f"Sampled data keys: {list(data.keys())}")
    
    return metadata, data

def validate_and_clean(metadata, data):
    # num samples metadata
    num_samples_metadata = metadata.shape[0]
    num_samples_data = data["X"].shape[0]
    
    if num_samples_metadata != num_samples_data:
        raise ValueError(f"Number of samples in metadata ({num_samples_metadata}) does not match number of samples in data ({num_samples_data}).")

    print("Sample sizes are consistent.")

    print("How many samples per label in both...")
    for i in range(data["y"].shape[1]):
        label_indices = np.where(data["y"][:, i] == 1)[0]
        print(f"\tLabel {i}: {len(label_indices)} samples in data, {len(metadata.iloc[label_indices])} samples in metadata.")
    

    if metadata.isnull().values.any():
        print("Metadata contains missing values. Filling with zeros.")
        metadata = metadata.fillna("Unknown")

    for key in data.keys():
        if np.any(np.isnan(data[key])):
            print(f"Data contains NaN values in {key}. Filling with zeros.")
            data[key] = np.nan_to_num(data[key])
    print("Data cleaned successfully.")
    return metadata, data

def exploratory_data_analysis(metadata, data):
    wd = pathlib.Path(__file__).parent.parent.parent.resolve()
    print(wd)
    media_path = wd / "media"
    # Summarize metadata
    print("metadata summary: ")
    print(metadata.describe())

    # visualize metadata (e.g., histogram for numeric columns)
    numeric_cols = metadata.select_dtypes(include=["number"]).columns
    print("Numeric columns: ", numeric_cols)
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(metadata[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(media_path / f"{col}_distribution.png")
    # Let's visualize the first 100 samples as for 256 characters, and the five features.
    # the five features will be ACTGN, but will be encoded as their one-hot encoding,
    # so from 0 to 4
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
    imin = 654
    imax = imin + 5
    for i in range(imin,imax,1):
        # black and white cmap
        print(data["widths"][i])
        axs[i-imin].imshow(data["X"][i, :, :].squeeze().T, aspect="auto", cmap="gray")
        axs[i-imin].set_title(f"sample {i}")
        axs[i-imin].set_ylabel("Sample Index")
        axs[i-imin].set_xlabel("Position")
    fig.savefig(media_path / "samples.png")
