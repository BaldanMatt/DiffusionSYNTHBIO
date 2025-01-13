from pathlib import Path
import gzip
import pandas as pd
import polars as pl
from scipy.io import mmread

DHS_INDEX_FILE = "DHS_Index_and_Vocabulary_hg38_WM20190703"
FDR01_hg38_FILE = "dat_bin_FDR01_hg38"
MAT_FDR01_hg38_FILE = "dat_bin_FDR01_hg38"

TXT_SUFFIX = "txt"
CSV_SUFFIX = "csv"
MM_SUFFIX = "mtx"
def main(data_dir):
    print("Starting main function...")
    print(f"\tThe data directory is located at: {data_dir}")
    filename = MAT_FDR01_hg38_FILE
    suffix = MM_SUFFIX
    
    if filename == DHS_INDEX_FILE:
        # We are reading the metadata file
        print("Reading DHS index file...")
        with gzip.open(data_dir / f"{filename}.{suffix}.gz", "rb") as f:
            file_content=f.readlines()
            # split each line into columns, tab separated
            file_content=[line.decode("utf-8").split("\t") for line in file_content]
            # Transform in a pandas dataframe
            file_content=pl.DataFrame(file_content)
            print("The file content is:\n", file_content)
    else:
        # We are reading the same data but binarized in a text format that is parsed more
        # efficiently because full of zeros.
        #
        # Sparse matrix data structure:
        # - Each stored elements has a triplet (row index, column index, value)
        # In thi case the values are only 1s because the zeros are not stored.
        if suffix == "mtx":
            print("Reading matrix text file...")
            with gzip.open(data_dir / f"{filename}.{suffix}.gz", "rb") as f:
                # Read the matrix market file
                matrix = mmread(f)
                print("The matrix is:\n", matrix)

        # We are reading the same data but in binarized text format that is huge if decoded
        elif suffix == "txt":
            print("Reading plain text file... large dimensions")
            with gzip.open(data_dir / f"{filename}.{suffix}.gz", "r") as f:
                pass
                # If wanted, we can implement a batch reading of the file and reconstruct the 
                # sparse matrix, but... it is already provided in .mtx format.

        pass


if __name__ == "__main__":
    # retrieve the current working directory and resolve its parent
    parent_dir = Path(__file__).resolve().parent.parent
    
    # init data directory
    data_dir = parent_dir / "data"

    main(data_dir)

