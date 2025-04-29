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
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eda import load_data as eda_load_data, validate_and_clean as eda_validate_and_clean
import umap
from sklearn.metrics import pairwise_distances

class Extracter:
    RANDOM_STATE = 42
    def __init__(self,
                 metadata_path,
                 data_path,
                 center: bool = True,
                 interactive: bool = True,
                 force: bool = False):
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.center = center
        self.interactive = interactive
        self.force = force

    def run(self, in_data: str = '', in_metadata: str = ''):
        print("Configuration...\n\tmetadata {}\n\tdata path {}\n\tcenter {}\n\tinteractive {}\n\tforce {}".format(
            self.metadata_path, self.data_path, self.center, self.interactive, self.force
        ))
        if self.do_eda_or_extract():
            self.eda()
        else:
            self.extract(in_data, in_metadata)

    def do_eda_or_extract(self) -> bool:
        if os.path.exists(self.metadata_path) and os.path.exists(self.data_path) and not self.force:
            print("Extracter has found the data... Doing EDA")
            return True
        else:
            print("Extracter has not found the data... Creating it")
            return False

    def eda(self):
        print("Starting EDA on data...")
        print(f"\tMetadata path: {metadata_path}")
        print(f"\tData path: {data_path}")
        plotter = Plotter()
        # Load metadata from CSV
        metadata, data = eda_load_data(metadata_path, data_path)
        # metadata, data = eda_validate_and_clean(metadata, data)
        # Summarize metadata
        # exploratory_data_analysis(metadata, data, interactive)

        # Extract metrics
        matches = mt.find_subsequence(data["X"], "TATA", "ACGTN")
        plotter.plot_find_subsequence(matches)
        # Reducer

        X_flat = data["X"].reshape(data["X"].shape[0], -1)
        dist_matrix = pairwise_distances(X_flat, metric="hamming")
        reducer = umap.UMAP(metric="precomputed", random_state = self.RANDOM_STATE)
        X_umap = reducer.fit_transform(dist_matrix)
        umap_df = pd.DataFrame(X_umap, columns=["UMAP_1", "UMAP_2"])
        umap_df = pd.concat([umap_df, metadata.reset_index(drop=True)], axis=1)
        plotter.plot_umap(umap_df, x="UMAP_1", y="UMAP_2", hue="component", palette="tab20")

        print("EDA terminated.")

    def extract(self, in_data: str | pathlib.Path, in_metadata:str | pathlib.Path):
        from src.utils.constants import DHS_metadata_schema
        data = load_data(in_data)
        meta = load_metadata(in_metadata, DHS_metadata_schema)
        print("Saving the metadata...")
        meta.write_csv(self.metadata_path)
        X = du.create_data(data, meta, center_in_summit=self.center, output_file=self.metadata_path)
        one_hot_x, one_hot_y, widths = du.parse_data(X)
        print("Saving one-hot encoded data... (sequences)")
        if self.center:
            filename_to_save = "DHS_one_hot_centered.npz"
        else:
            filename_to_save = "DHS_one_hot.npz"
        # Saving the extracted one coded sequences
        # remove the last part after the last / in seqspath
        np.savez_compressed(
            self.data_path,
            X=one_hot_x,
            y=one_hot_y,
            widths=widths
        )
        return one_hot_x, one_hot_y, widths

class Plotter:
    def __init__(self,
                 style: str = "seaborn-v0_8-darkgrid",
                 figsize: tuple = (10,6),
                 show: bool = True,):
        plt.style.use(style)
        self.figsize=figsize
        self.show=show

    def _finalize_plot(self):
        if self.show:
            plt.show()
        else:
            if save_path is not None:
                plt.savefig(save_path)
            else:
                print("Save path is Not provided!")
    def plot_find_subsequence(self, matches, hue=None, title='', xlabel='', ylabel='', save_path = None):
        plt.Figure(figsize=self.figsize)
        frequencies = matches.sum(axis=0)
        sns.lineplot(frequencies, hue=hue, ax=plt.gca())
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        self._finalize_plot()

    def plot_umap(self, umap_df, x="", y="", palette="", hue = None, title='', xlabel='', ylabel='', save_path = None):
        plt.Figure(figsize=self.figsize)
        sns.scatterplot(umap_df, x=x, y=y, hue=hue, palette=palette, ax=plt.gca())
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        self._finalize_plot()

def parse_argument():
    parser = argparse.ArgumentParser(description="Experiment for DHS sequences")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--center", action="store_true", help="Center the sequences in the summit of the DHS")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--force", action="store_true", help="Force the experiment to run even if data exists")
    args = parser.parse_args()
    return args.metadata_path, args.data_path, args.center, args.interactive, args.force

if __name__ == "__main__":
    metadata_path, data_path, center, interactive, force = parse_argument()
    print("Running experiment with: \n\t metadata path {}\n\t data path {}\n\t center {}\n\t interactive {}\n\t force {}".format(
        metadata_path,data_path, center, interactive, force))
    extracter = Extracter(metadata_path, data_path, center, interactive, force)

    project_dir = pathlib.Path(__file__).resolve().parent
    in_data_path = project_dir / "data" / "dat_bin_FDR01_hg38.mtx.gz"
    in_metadata_path = project_dir / "data" / "DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
    extracter.run(in_data_path, in_metadata_path)

