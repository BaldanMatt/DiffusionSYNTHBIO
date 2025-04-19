import pathlib, os
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import src.utils.metrics as mt
import src.utils.data_utils as du
from src.utils.load_data import load_data, load_metadata
import polars as pl
import numpy as np

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
        X = du.create_data(data, meta, center_in_summit=center_in_summit)
        X.write_csv(seqspath)
    one_hot_x, one_hot_y, widths = du.parse_data(X)
    print("Saving one-hot encoded data... (sequences)")
    if center_in_summit:
        filename_to_save = "DHS_one_hot_centered.npz"
    else:
        filename_to_save = "DHS_one_hot.npz"

    # Saving the extracted one coded sequences
    np.savez_compressed(
        datadir / filename_to_save,
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
    retrieve_DHS_sequences_from_hg38(seqspath=resD / "DHS_extracted_seqs.csv",
                                     datadir=dataD,
                                     datafile="dat_bin_FDR01_hg38.mtx.gz",
                                     metafile="DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz",
                                     center_in_summit=False
                                     )

if __name__ == "__main__":
    main()

