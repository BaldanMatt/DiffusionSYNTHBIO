import numpy as np
from pathlib import Path
import sys
import os
sys.path.insert(0, Path(os.getcwd()) / "src")


from src.utils import parse_data, create_data
from src.utils import load_data, load_metadata
import argparse
import polars as pl
from src.utils.constants import DHS_metadata_schema


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Parse data")
    parser.add_argument("--data_dir_path", type=str, help="Path to data file")
    parser.add_argument("--data_file_name", type=str, help="Path to data file")
    parser.add_argument("--metadata_file_name", type=str, help="Path to metadata file")
    parser.add_argument("--n_regions", type=int, help="Number of regions to extract")
    args = parser.parse_args()
    args.data_dir_path = Path(args.data_dir_path)
    return args


def test_parsing():
    print("Testing parsing...")
    args = parse_command_line_arguments()
    data = load_data(
        args.data_dir_path,
        args.data_file_name,
    )
    metadata = load_metadata(
        args.data_dir_path, args.metadata_file_name, DHS_metadata_schema
    )

    # Test create_data to query genome hg38
    if os.path.exists(args.data_dir_path / "DHS_extracted_seqs.csv"):
        print("Extracted seqs already exist. Loading them...")
        extracted_seqs = pl.read_csv(args.data_dir_path / "DHS_extracted_seqs.csv")
    else:
        extracted_seqs = create_data(data, metadata, args.n_regions)

    print("Creating data passed.", extracted_seqs)
    # Test parse_data to convert data to numpy arrays
    ## Testing both with read data and with extracted data
    print("Starting parsing test...")
    one_hot_x, one_hot_y, widths = parse_data(data)
    print("Parsing passed.")
    print("Starting parsing test with extracted data...")
    t_one_hot_x, t_one_hot_y, t_widths = parse_data(extracted_seqs)
    print("Parsing test passed.")

    print("saving such hot labels and extracted seqs is csv files...")
    # Save extracted seqs as a csv file
    extracted_seqs.write_csv(args.data_dir_path / "DHS_extracted_seqs.csv")

    # Save one hot labels as tensor objects
    print(f"type of X is {type(t_one_hot_x)} and shape is {t_one_hot_x.shape}")
    print(f"type of labels is {type(t_one_hot_y)} and shape is {t_one_hot_y.shape}")
    print(f"type of t_widths is {type(t_widths)} and shape is {t_widths.shape}")

    np.savez_compressed(
        args.data_dir_path / "dataset_compressed.npz",
        data=t_one_hot_x,
        labels=t_one_hot_y,
        widths=t_widths,
    )


if __name__ == "__main__":
    test_parsing()
