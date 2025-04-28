import pathlib, os
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from analytics.utils.eda import exploratory_data_analysis
import src.utils.metrics as mt
import src.utils.data_utils as du
from src.utils.load_data import load_data, load_metadata
import polars as pl
import numpy as np
import pandas as pd
import argparse

from utils.eda import load_data as eda_load_data, validate_and_clean as eda_validate_and_clean

def parse_argument():
    parser = argparse.ArgumentParser(description="Experiment for DHS sequences")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--force", action="store_true", help="Force the experiment to run even if data exists")
    args = parser.parse_args()
    return args.metadata_path, args.data_path, args.force


def eda_on_data(metadata_path,
                data_path):
    print("Starting EDA on data...")
    print(f"\tMetadata path: {metadata_path}")
    print(f"\tData path: {data_path}")
    # Load metadata from CSV
    metadata, data = eda_load_data(metadata_path, data_path)
    metadata, data = eda_validate_and_clean(metadata, data)
    # Summarize metadata 
    exploratory_data_analysis(metadata, data)
    print("EDA terminated.")


def retrieve_DHS_sequences_from_hg38(seqspath: pathlib.Path = None,
                                     datadir: pathlib.Path = None,
                                     datafile: str = None,
                                     metafile: str = None,
                                     center_in_summit: bool = True
                                     ):

    if os.path.exists(seqspath):
        print(f"Data path {seqspath} exists.")
        X = pl.read_csv(seqspath)
    else:
        from src.utils.constants import DHS_metadata_schema
        data = load_data(datadir, datafile)
        meta = load_metadata(datadir, metafile, DHS_metadata_schema)
        X = du.create_data(data, meta, center_in_summit=center_in_summit, output_file=seqspath)
    one_hot_x, one_hot_y, widths = du.parse_data(X)
    print("Saving one-hot encoded data... (sequences)")
    if center_in_summit:
        filename_to_save = "DHS_one_hot_centered.npz"
    else:
        filename_to_save = "DHS_one_hot.npz"

    # Saving the extracted one coded sequences
    # remove the last part after the last / in seqspath
    seqspath = seqspath.parent
    np.savez_compressed(
        seqspath / filename_to_save,
        X=one_hot_x,
        y=one_hot_y,
        widths=widths
    )
    print(X)
    return one_hot_x, one_hot_y, widths
def sample_experiment():
    pass


def main():
    wd = pathlib.Path.cwd()
    print(f"We are currently working in {wd}")
    
    expD = pathlib.Path(__file__).parent
    print(f"Experiment directory is in {expD}")

    dataD = wd / "data"
    resD = wd / "results"

    print("Retrieving DHS sequences...")
    retrieve_DHS_sequences_from_hg38(seqspath=resD / "DHS_extracted_seqs_centered.csv",
datadir=dataD,
                                     datafile="dat_bin_FDR01_hg38.mtx.gz",
                                     metafile="DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz",
                                     center_in_summit=True
                                     )
    

def check_experiment_result(data_dir, result_dir):
    print(f"Checking all results in results directory {result_dir}") 
    # Check if the result directory exists
    if not os.path.exists(result_dir):
        print(f"Result directory {result_dir} does not exist.")
        return
    # Check if the result directory is empty
    if not os.listdir(result_dir):
        print(f"Result directory {result_dir} is empty.")
        return
    # Check if the result directory contains expected files
    print(os.listdir(result_dir))
    
    center_in_summit = False
    # Load Centered results
    print("Loading not centered results...") if center_in_summit else print("Loading centered results...")
    filename = "DHS_extracted_seqs_centered.csv" if center_in_summit else "DHS_extracted_seqs.csv"
    metadata = pd.read_csv(result_dir / filename)
    # Load the one-hot encoded data
    print("Loading one-hot encoded data...")
    datafilename = "DHS_one_hot_centered.npz" if center_in_summit else "DHS_one_hot.npz"
    datafilename = "generated.npz"
    if center_in_summit:
        data = np.load(data_dir / datafilename)
    else:
        data = np.load(data_dir / datafilename)
    print(f"Data shape: {metadata.shape}")
    print(f"Data columns: {metadata.columns}")
    print(f"Data info: {metadata.info()}")
    print(f"Data describe: {metadata.describe()}")
    
    from src.utils.metrics import motif_correlation
    samples_per_label = 10000 
    num_labels = data["labels"].shape[1]
    num_labels = data["labels"].shape[1]
    motif = "TATAWAW"
    import matplotlib.pyplot as plt
    # Prepare the plot
    fig, ax = plt.subplots(num_labels, 1, figsize=(10, 5 * num_labels), sharex=True)
    fig2, ax2 = plt.subplots(num_labels, 1, figsize=(10, 5 * num_labels), sharex=True)
    sampled_indices = []
    for label in range(num_labels):
        print(f"Label {label}")
        # Get the indices of the samples with the current label
        label_indices = np.where(data["labels"][:, label] == 1)[0]
        
        # Sample the specified number of indices
        sampled = np.random.choice(label_indices, size=samples_per_label, replace=False)
        sampled_indices.extend(sampled)
        
        # Get the sampled data
        sampled_data = data["data"][sampled]
        
        # Calculate the frequency distribution for the motif
        distr = find_subsequence(sampled_data, motif, "ACTGN").mean(axis=1)
        distr_2 = find_subsequence(sampled_data, motif, "ACTGN").mean(axis=0)
        # Plot the frequency distribution
        if num_labels > 1:
            ax[label].plot(distr, label=f"Label {label}")
            ax2[label].plot(distr_2, label=f"Label {label}")
            ax[label].set_title(f"Frequency Distribution for Label {label}")
            ax2[label].set_title(f"Frequency Distribution for Label {label}")
            ax[label].set_xlabel("Position")
            ax2[label].set_xlabel("Position")
            ax[label].set_ylabel("Frequency")
            ax2[label].set_ylabel("Frequency")
            ax[label].legend()
            ax2[label].legend()
        else:
            ax.plot(distr, label=f"Label {label}")
            ax2.plot(distr_2, label=f"Label {label}")
            ax.set_title(f"Frequency Distribution for Label {label}")
            ax2.set_title(f"Frequency Distribution for Label {label}")
            ax.set_xlabel("Position")
            ax2.set_xlabel("Position")
            ax.set_ylabel("Frequency")
            ax2.set_ylabel("Frequency")
            ax.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    
    pass


if __name__ == "__main__":
    metadata_path, data_path, force = parse_argument()
    if os.path.exists(metadata_path) and os.path.exists(data_path) and not force:
        print("Metadata and data paths exist.")
        eda_on_data(metadata_path, data_path)
    else:
        main()
        #check_experiment_result(pathlib.Path.cwd() / "data", pathlib.Path.cwd() / "results")

