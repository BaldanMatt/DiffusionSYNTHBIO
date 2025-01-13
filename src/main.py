from pathlib import Path
import gzip
import pandas as pd
import polars as pl
DHS_INDEX_FILE = "DHS_Index_and_Vocabulary_hg38_WM20190703"
FDR01_hg38_FILE = "dat_bin_FDR01_hg38"
TXT_SUFFIX = "txt"
CSV_SUFFIX = "csv"
def main(data_dir):
    print("Starting main function...")
    print(f"\tThe data directory is located at: {data_dir}")
    filename = FDR01_hg38_FILE
    suffix = TXT_SUFFIX
    with gzip.open(data_dir / f"{filename}.{suffix}.gz", "r") as f:
        file_content=f.readlines()
        # split each line into columns, tab separated
        file_content=[line.decode("utf-8").split("\t") for line in file_content]
        # Transform in a pandas dataframe
        file_content=pl.DataFrame(file_content)
        print("The file content is:\n", file_content)

    pass


if __name__ == "__main__":
    # retrieve the current working directory and resolve its parent
    parent_dir = Path(__file__).resolve().parent.parent
    
    # init data directory
    data_dir = parent_dir / "data"

    main(data_dir)

