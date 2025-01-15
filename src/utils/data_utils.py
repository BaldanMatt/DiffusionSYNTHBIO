import polars as pl
import numpy as np
from pathlib import Path
from Bio import SeqIO
import torch as pt
# We need to import the Bio.io to read fasta files
from utils.download_hg38_genome import download_hg38_genome_or_load

def create_data(data: pl.DataFrame, metadata: pl.DataFrame, n_regions: int = 10, len_seq = 256):
    print("Parsing data...")
    # We need to download the human genome
    genome_index = download_hg38_genome_or_load()
    # We need to read the genome based on the content of metadata
    # in metadata i have three columns (seqname, start, end)
    # for each row in metadata we need to extract the sequence from the genome
    extracted_seq = {"region_name": [],
                     "seqname": [],
                     "start": [],
                     "end": [],
                     "raw_sequence": [],
                     "DHS_width": [],
                     "component": []}
    selected_records = metadata.select(["seqname", "start", "end","component"]).head(n_regions)
    for row in selected_records.iter_rows(): 
        seqname, start, end, component = row
        print(f"Extracting sequence {seqname} from {start} to {end}")
        print(f"Types are: {type(seqname)} {type(start)} {type(end)}")
        if seqname not in genome_index:
            raise ValueError(f"The sequence {seqname} is not present in the genome.")
        seq_record = genome_index[seqname].seq[start:end] 
        region_name=f"{seqname}:{start}-{end}"
        extracted_seq["region_name"].append(region_name)
        extracted_seq["seqname"].append(seqname)
        extracted_seq["start"].append(start)
        extracted_seq["end"].append(end)
        if len(seq_record) < len_seq:
            seq_record = seq_record + "N"*(len_seq-len(seq_record))
        elif len(seq_record) > len_seq:
            seq_record = seq_record[:len_seq]
        extracted_seq["raw_sequence"].append(str(seq_record).upper())
        extracted_seq["DHS_width"].append(end-start)
        extracted_seq["component"].append(component)

    extracted_seq = pl.from_dict(
        extracted_seq
    )
    return extracted_seq

def one_hot_encode(data):
    sequence_length = len(data[0])
    data = np.ascontiguousarray(data)
    chars = data.view("S1").reshape(-1, sequence_length, 4)[..., 0]
    masks = [chars == b"A", chars == b"C", chars == b"G", chars == b"T"]
    nums = np.select(masks, [0,1,2,3], default=4)
    one_hot = np.eye(5)[nums]
    return one_hot

def one_hot_encode_labels(labels):
    masks = [labels == val for val in np.unique(labels)]
    nums = np.select(masks, list(range(len(masks))))
    one_hot = np.eye(len(masks))[nums]
    return one_hot

def check_one_hot_encode(data, one_hot, only_first_n_entries=None):
    nums = np.select(one_hot.T.astype(bool), [0,1,2,3,4]).T
    chars = np.array([b"A", b"C", b"G", b"T", b"N"])[nums]
    for i, (recon, row) in enumerate(zip(chars, data)):
        if only_first_n_entries is not None and i >= only_first_n_entries:
            break
        recon = "".join(recon.astype(str))
        if row != recon:
            return False
    return True

def parse_data(data: pl.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    column_subset = ["raw_sequence","DHS_width","component"]
    # I need to convert raw_sequence in an actual sequence of character that i can also binarize


    data = data.select(column_subset)
    X = data["raw_sequence"].to_numpy().astype(str)
    labels = data["component"].to_numpy().astype(str)
    width = data["DHS_width"].to_numpy().astype(np.int64)
    # Log before one hot
    print("data before one_hot: ", X.shape, X.dtype)
    print("labels before one_hot: ", labels.shape, labels.dtype)
    one_hot = one_hot_encode(X)
    one_hot_labels = one_hot_encode_labels(labels)
    # Log after one hot
    print("one_hot: ", one_hot.shape, one_hot.dtype)
    print("one_hot_labels: ", one_hot_labels.shape, one_hot_labels.dtype)
    print("is_correct ", check_one_hot_encode(X, one_hot, only_first_n_entries=150))
    
    return one_hot, one_hot_labels, width

if __name__ == "__main__":
    parse_data()

