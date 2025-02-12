import polars as pl
import numpy as np
from pathlib import Path
from Bio import SeqIO

# We need to import the Bio.io to read fasta files
from utils.download_hg38_genome import download_hg38_genome_or_load
from tqdm import tqdm


def create_data(
    data: pl.DataFrame, metadata: pl.DataFrame, n_regions: int = None, len_seq=256
):
    print("Parsing data...")
    if n_regions is None:
        n_regions = metadata.shape[0]
    # We need to download the human genome
    genome_index = download_hg38_genome_or_load()
    # We need to read the genome based on the content of metadata
    # in metadata i have three columns (seqname, start, end)
    # for each row in metadata we need to extract the sequence from the genome
    extracted_seq = {
        "region_name": [],
        "seqname": [],
        "start": [],
        "end": [],
        "raw_sequence": [],
        "DHS_width": [],
        "component": [],
    }
    selected_records = metadata.select(["seqname", "start", "end", "component"]).head(n_regions)

    for row in tqdm(selected_records.iter_rows(), desc="Extracting sequences..."):
        seqname, start, end, component = row
        # print(f"Extracting sequence {seqname} from {start} to {end}")
        # print(f"Types are: {type(seqname)} {type(start)} {type(end)}")
        # tqdm.write(f"Extracting sequence {seqname} from {start} to {end} (component: {component})")
        if seqname not in genome_index:
            raise ValueError(f"The sequence {seqname} is not present in the genome.")
        seq_record = genome_index[seqname].seq[start:end]
        region_name = f"{seqname}:{start}-{end}"
        extracted_seq["region_name"].append(region_name)
        extracted_seq["seqname"].append(seqname)
        extracted_seq["start"].append(start)
        extracted_seq["end"].append(end)
        if len(seq_record) < len_seq:
            seq_record = seq_record + "N" * (len_seq - len(seq_record))
        elif len(seq_record) > len_seq:
            seq_record = seq_record[:len_seq]
        extracted_seq["raw_sequence"].append(str(seq_record).upper())
        extracted_seq["DHS_width"].append(end - start)
        extracted_seq["component"].append(component)

    extracted_seq = pl.from_dict(extracted_seq)
    return extracted_seq


def one_hot_encode(data, batch_size=100000):
    sequence_length = len(data[0])
    num_batches = int(np.ceil(len(data) / batch_size))
    total_rows = len(data)
    one_hot_encoded_torch = torch.zeros((total_rows, sequence_length, 5), dtype=torch.bool)
    print("num_batches: ", num_batches, " total_rows: ", total_rows, " starting batch...")
    for i in tqdm(range(num_batches)):
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        batch_data = np.ascontiguousarray(batch_data)
        chars = batch_data.view("S1").reshape(-1, sequence_length, 4)[..., 0]
        masks = [chars == b"A", chars == b"C", chars == b"G", chars == b"T"]
        nums = np.select(masks, [0, 1, 2, 3], default=4)
        one_hot = np.eye(5, dtype=bool)[nums]
        one_hot_encoded[i * batch_size : (i + 1) * batch_size] = one_hot
    return one_hot_encoded


def one_hot_encode_labels(labels):
    masks = [labels == val for val in np.unique(labels)]
    nums = np.select(masks, list(range(len(masks))))
    one_hot = np.eye(len(masks), dtype=bool)[nums]
    return one_hot


def check_one_hot_encode(data, one_hot, only_first_n_entries=None):
    nums = np.select(one_hot.T.astype(bool), [0, 1, 2, 3, 4]).T
    chars = np.array([b"A", b"C", b"G", b"T", b"N"])[nums]
    for i, (recon, row) in enumerate(zip(chars, data)):
        if only_first_n_entries is not None and i >= only_first_n_entries:
            break
        recon = "".join(recon.astype(str))
        if row != recon:
            return False
    return True


def check_one_hot(data, one_hot, only_first_n_entries=None):
    print("debug, data: ", data.shape, data.dtype, type(data))
    print("debug, one_hot: ", one_hot.shape, one_hot.dtype, type(one_hot))
    nums = np.argmax(one_hot, axis=-1)
    chars = np.array([ord(c) for c in "ACGTN"], dtype=np.uint8)[nums]
    for i, (recon, row) in enumerate(zip(chars, data)):
        if only_first_n_entries is not None and i >= only_first_n_entries:
            break
        recon = "".join(chr(c) for c in recon.tolist())
        if row != recon:
            return False
    return True


def parse_data(data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    column_subset = ["raw_sequence", "DHS_width", "component"]
    # I need to convert raw_sequence in an actual sequence of character that i can also binarize
    data = data.select(column_subset)
    X = data["raw_sequence"].to_numpy().astype(str)
    labels = data["component"].to_numpy().astype(str)
    width = data["DHS_width"].to_numpy().astype(np.int64)
    # Log before one hot
    print("data before one_hot: ", X.shape, X.dtype)
    print("labels before one_hot: ", labels.shape, labels.dtype)
    print("Running one hot encoding... on X")
    one_hot_x = one_hot_encode(X)
    print("Running one hot encoding... on labels")
    one_hot_y = one_hot_encode_labels(labels)
    # Log after one hot
    print(f"one_hot: {type(one_hot_x)}, {one_hot_x.shape}, {one_hot_x.dtype}")
    print(f"one_hot_labels: {type(one_hot_y)}, {one_hot_y.shape}, {one_hot_y.dtype}")
    # Check if the one hot encoding is correct
    print("Running check_one_hot_encode...")
    print("is_correct ", check_one_hot(X, one_hot_x, only_first_n_entries=1000))

    return one_hot_x, one_hot_y, width


if __name__ == "__main__":
    pass
