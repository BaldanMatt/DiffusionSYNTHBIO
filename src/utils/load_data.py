from pathlib import Path
import gzip
import pandas as pd
import polars as pl
from polars import schema
from scipy.io import mmread
from utils.constants import DHS_metadata_schema, BIOSAMPLES_metadata_schema


DHS_metadata = "DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
BIOSAMPLES_metadata = "DHS_Index_and_Vocabulary_metadata.tsv"
FDR01_hg38_FILE = "dat_bin_FDR01_hg38.txt.gz"
MAT_FDR01_hg38_FILE = "dat_bin_FDR01_hg38.mtx.gz"
SIGNAL_MAT_FILE="dat_FDR01_hg38.txt.gz"
TRAIN_LIGHT = "train_all_classifier_light.csv.gz"
TEST_LIGHT = "test_all_classifier_light.csv.gz"
VAL_LIGHT = "validation_all_classifier_light.csv.gz"

# LOW MEMORY USAGE
BATCH_LINES_SIZE = int(1e5)
from itertools import islice

def read_gz_in_batches(f, batch_size=BATCH_LINES_SIZE):
    while True:
        batch_lines = list(islice(f, batch_size))
        if not batch_lines:
            break
        for line in batch_lines:
            yield line

def load_data(data_dir, filename: str):
    print(f"The data directory is located at: {data_dir}")
    print(f"\tThe filename is: {filename}")
    file_pieces = filename.split(".")
    if len(file_pieces) == 2:
        filename, suffix_format = file_pieces
        suffix_gz = None
    elif len(file_pieces) == 3:
        filename, suffix_format, suffix_gz = file_pieces
    else:
        raise ValueError("The filename is not in the expected format, too many dots (.)")

    if suffix_gz is not None:
        full_suffix = f"{suffix_format}.{suffix_gz}"
        filename = f"{filename}.{full_suffix}"
        print("The full suffix is:", full_suffix)
        if suffix_format == "txt":
            pass
        elif suffix_format == "mtx":
           # We are reading the same data but binarized in a text format that is parsed more
            # efficiently because full of zeros.
            #
            # Sparse matrix data structure:
            # - Each stored elements has a triplet (row index, column index, value)
            # Ins thi case the values are only 1s because the zeros are not stored.
            with gzip.open(data_dir / f"{filename}", "rb") as f:
                print("Reading matrix text file...")
                file_content = mmread(f)
        elif suffix_format == "csv":
            print("Reading csv file...")
            file_content = pl.read_csv(data_dir / f"{filename}", separator="\t", has_header=True)
    else:
        raise ValueError("The suffix format is not recognized")
    print("The file content is:\n", file_content)
    print("The file content schema is:\n", file_content.schema) if isinstance(file_content, pl.DataFrame) else None
    return file_content

def load_metadata(data_dir, filename: str,polar_schema = None):
    print(f"The data directory is located at: {data_dir}")
    print(f"\tThe filename is: {filename}") 
    file_pieces = filename.split(".")
    if len(file_pieces) == 2:
        filename, suffix_format = file_pieces
        suffix_gz = None
    elif len(file_pieces) == 3:
        filename, suffix_format, suffix_gz = file_pieces
    else:
        raise ValueError("The filename is not in the expected format, too many dots (.)")
    
    if suffix_gz is not None:
        full_suffix = f"{suffix_format}.{suffix_gz}"
        filename = f"{filename}.{full_suffix}"
        if suffix_format == "txt":
            # We are reading the same data but in binarized text format that is huge if decoded
            print("Reading plain text file...")
             
            if polar_schema is None:
                print("[WARNING] you asked to use polar but you have not provided the schema, it will be slow and it will save all columns to pl.String()")
                file_content = pl.read_csv(data_dir / f"{filename}",
                                           infer_schema=False,
                                           null_values=["NA"],
                                           separator="\t",
                                           has_header=True,
                                           n_rows=5)
            else:
                file_content = pl.read_csv(data_dir / f"{filename}",
                                            schema=polar_schema,
                                            infer_schema_length=int(1e5),
                                            null_values=["NA"],
                                            separator="\t",
                                            has_header=True)

        elif suffix_format == "csv":
            print("Reading csv file...")
            pass
        elif suffix_format == "tsv":
            print("Reading tsv file...")
            full_suffix = suffix_format
            filename = f"{filename}.{full_suffix}"
            if polar_schema is None:
                print("[WARNING] you asked to use polar but you have not provided the schema, it will be slow and it will save all columns to pl.String()")
                file_content = pl.read_csv(data_dir / f"{filename}",
                                           infer_schema=False,
                                           null_values=["NA"],
                                           separator="\t",
                                           has_header=True)
            else:
                file_content = pl.read_csv(data_dir / f"{filename}",
                                            schema=polar_schema,
                                            infer_schema_length=int(1e5),
                                            null_values=["NA"],
                                            separator="\t",
                                            has_header=True)
        else:
            raise ValueError("The suffix format is not recognized")
    else:
        print("Reading uncompressed tsv file...")
        full_suffix = suffix_format
        filename = f"{filename}.{full_suffix}"
        if polar_schema is None:
            print("[WARNING] you asked to use polar but you have not provided the schema, it will be slow and it will save all columns to pl.String()")
            file_content = pl.read_csv(data_dir / f"{filename}",
                                       infer_schema=False,
                                       null_values=["NA"],
                                       separator="\t",
                                       has_header=True)
        else:
            file_content = pl.read_csv(data_dir / f"{filename}",
                                        schema=polar_schema,
                                        infer_schema_length=int(1e5),
                                        null_values=["NA"],
                                        separator="\t",
                                        has_header=True)

    print("The file content is:\n", file_content)
    print("The file content schema is:\n", file_content.schema) if isinstance(file_content, pl.DataFrame) else None
    return file_content

if __name__ == "__main__":
    # retrieve the current working directory and resolve its parent
    pparent_dir = Path(__file__).resolve().parent.parent
    
    # init data directory
    data_dir = parent_dir / "data"
    # test DHS metadata
    filename = DHS_metadata
    file_content=load_metadata(data_dir, filename,polar_schema=DHS_metadata_schema)
    # test bio samples metadata
    filename = BIOSAMPLES_metadata
    file_content=load_metadata(data_dir, filename,polar_schema=BIOSAMPLES_metadata_schema)
    # test Train file
    filename = TRAIN_LIGHT
    file_content=load(data_dir, filename,)
    # test Test file
    filename = TEST_LIGHT
    file_content=load(data_dir, filename,)
    # test Val file
    filename = VAL_LIGHT
    file_content=load(data_dir, filename,)
