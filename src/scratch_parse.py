from utils import parse_data, create_data
from utils import load_data, load_metadata
from pathlib import Path
import argparse
import torch
import pandas as pd
import os
from utils.constants import DHS_metadata_schema

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
    data = load_data(args.data_dir_path, args.data_file_name, )
    metadata = load_metadata(args.data_dir_path, args.metadata_file_name, DHS_metadata_schema)

    # Test create_data to query genome hg38
    if os.path.exists(args.data_dir_path / "DHS_extracted_seqs.csv"):
        print("Extracted seqs already exist. Loading them...")
        extracted_seqs = pd.read_csv(args.data_dir_path / "DHS_extracted_seqs.csv")
    else:
        extracted_seqs = create_data(data, metadata, args.n_regions)
    print("Creating data passed.", extracted_seqs)
    # Test parse_data to convert data to numpy arrays
    ## Testing both with read data and with extracted data
    print("Starting parsing test...")
    one_hot_x, one_hot_labels, widths = parse_data(data)
    print("Parsing passed.")
    print("Starting parsing test with extracted data...")
    t_one_hot_x, t_one_hot_labels, t_widths = parse_data(extracted_seqs)
    print("Parsing test passed.")    

    print("saving such hot labels and extracted seqs is csv files...")
    # Save extracted seqs as a csv file
    extracted_seqs.write_csv(args.data_dir_path / "DHS_extracted_seqs.csv") 

    # Save one hot labels as tensor objects
    print(f"type of t_one_hot_x is {type(t_one_hot_x)} and shape is {t_one_hot_x.shape}")
    print(f"type of t_one_hot_labels is {type(t_one_hot_labels)} and shape is {t_one_hot_labels.shape}")
    print(f"type of t_widths is {type(t_widths)} and shape is {t_widths.shape}")
    torch.save(t_one_hot_x, args.data_dir_path / "one_hot_x.pt")
    torch.save(t_one_hot_labels, args.data_dir_path / "one_hot_labels.pt")
    torch.save(t_widths, args.data_dir_path / "widths.pt")

if __name__ == "__main__":
    test_parsing()
