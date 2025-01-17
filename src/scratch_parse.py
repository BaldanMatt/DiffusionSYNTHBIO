from utils import parse_data, create_data
from utils import load_data, load_metadata
from pathlib import Path
import argparse

from utils.constants import DHS_metadata_schema

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Parse data")
    parser.add_argument("--data_dir_path", type=str, help="Path to data file")
    parser.add_argument("--data_file_name", type=str, help="Path to data file")
    parser.add_argument("--metadata_file_name", type=str, help="Path to metadata file")
    args = parser.parse_args()
    args.data_dir_path = Path(args.data_dir_path)
    return args

def test_parsing():
    print("Testing parsing...")
    args = parse_command_line_arguments()
    data = load_data(args.data_dir_path, args.data_file_name, )
    metadata = load_metadata(args.data_dir_path, args.metadata_file_name, DHS_metadata_schema)

    # Test create_data to query genome hg38
    extracted_seqs = create_data(data, metadata, n_regions=500)
    print("Creating data passed.", extracted_seqs)
    # Test parse_data to convert data to numpy arrays
    ## Testing both with read data and with extracted data
    one_hot_x, one_hot_labels, widths = parse_data(data)
    one_hot_x_new, one_hot_labels_new, widths_new = parse_data(extracted_seqs)
    print("Parsing test passed.")    

if __name__ == "__main__":
    test_parsing()