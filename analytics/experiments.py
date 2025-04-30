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
                 vocabulary_path,
                 dhs_by_biosample_path,
                 vocabulary_metadata_path,
                 biosamples: list = None,
                 center: bool = True,
                 interactive: bool = True,
                 force: bool = False):
        self.vocabulary_path = vocabulary_path
        self.dhs_by_biosample_path = dhs_by_biosample_path
        self.vocabulary_metadata_path = vocabulary_metadata_path
        self.res_dhs_by_biosample_meta_name: str | None = None
        self.res_dhs_by_biosample_one_hot_name: str | None = None
        self.biosamples = biosamples
        self.center = center
        self.interactive = interactive
        self.force = force

    def run(self):
        print("Configuration...\n\tvocabulary path {}\n\tdhs by biosample path {}\n\tvocabulary metadata path {}\n\tbiosamples {}\n\tcenter {}\n\tinteractive {}\n\tforce {}".format(
            self.vocabulary_path,
            self.dhs_by_biosample_path,
            self.vocabulary_metadata_path,
            self.biosamples,
            self.center, self.interactive, self.force
        ))
        if self.biosamples:
            res_dhs_by_biosample_one_hot_name = 'DHS_one_hot_samples'
            res_dhs_by_biosample_meta_name = 'DHS_sequences_samples'
            for biosample in self.biosamples:
                # Build file name concatenating the biosamples
                res_dhs_by_biosample_one_hot_name = res_dhs_by_biosample_one_hot_name + "_{}".format(biosample)
                res_dhs_by_biosample_meta_name = res_dhs_by_biosample_meta_name + "_{}".format(biosample)

            path_dirs = str(pathlib.Path(__file__).parent.parent)
            self.res_dhs_by_biosample_one_hot_name = path_dirs + "/" + res_dhs_by_biosample_one_hot_name
            self.res_dhs_by_biosample_meta_name = path_dirs + "/" + res_dhs_by_biosample_meta_name
        else:
            self.res_dhs_by_biosample_one_hot_name = "DHS_one_hot"
            self.res_dhs_by_biosample_meta_name = "DHS_sequences"
        if self.center:
            self.res_dhs_by_biosample_meta_name += "_centered"
            self.res_dhs_by_biosample_one_hot_name += "_centered"
        print("We are going to process {} with metadata {}".format(self.res_dhs_by_biosample_one_hot_name, self.res_dhs_by_biosample_meta_name))
        if self.do_eda_or_extract():
            self.eda()
        else:
            self.extract()

    def do_eda_or_extract(self) -> bool:
        if os.path.exists(self.res_dhs_by_biosample_meta_name) and os.path.exists(self.res_dhs_by_biosample_one_hot_name) and not self.force:
            print("Extracter has found the data... Doing EDA")
            return True
        else:
            print("Extracter has not found the data... Creating it")
            return False

    def eda(self):
        print("Starting EDA on data...")
        print(f"\tMetadata path: {self.res_dhs_by_biosample_meta_name}")
        print(f"\tData path: {self.res_dhs_by_biosample_one_hot_name}")
        plotter = Plotter()
        # Load metadata from CSV
        metadata, data = eda_load_data(self.res_dhs_by_biosample_meta_name, self.res_dhs_by_biosample_one_hot_name)
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

    def extract(self):
        from src.utils.constants import DHS_metadata_schema
        data = load_data(self.dhs_by_biosample_path)
        vocabulary = load_metadata(self.vocabulary_path, DHS_metadata_schema)
        metadata = pd.read_csv(self.vocabulary_metadata_path, sep="\t", index_col=0, dtype={'library order':np.int32})
        print(metadata)
        if self.biosamples:
            # We find the indexes of the DHS accessible in the biosamples from
            biosample_indexes = metadata[metadata["Biosample name"].isin(self.biosamples)].index
            print("We are keeping {} libraries from {} samples".format(len(biosample_indexes), self.biosamples))
            # Now we need to keep the data that are accessible in those biosamples
            data = data[:, biosample_indexes]
            # Now we need to find the indexes of the DHS that we are accessible at least one in those Biosamples
            row_indexes = data.getnnz(axis=1) > 0
            dhs_in_biosample_indexes = np.where(row_indexes)[0]
            print("We are keeping {} DHS from {} samples".format(len(dhs_in_biosample_indexes), self.biosamples))
            # Now we need to keep only the metadata of the DHS that are accessible in those biosamples
            vocabulary = vocabulary[dhs_in_biosample_indexes]
            input()
        # Create data already save the vocabulary res file
        X = du.create_data(vocabulary, center_in_summit=self.center, output_file=self.res_dhs_by_biosample_meta_name)
        one_hot_x, one_hot_y, widths = du.parse_data(X)
        print("Saving one-hot encoded data... (sequences)")

        # Saving the extracted one coded sequences
        # remove the last part after the last / in seqspath
        np.savez_compressed(
            self.res_dhs_by_biosample_one_hot_name +".npz",
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
    parser.add_argument("--vocabulary_path", type=str, required=True, help="Path to the metadata file")
    parser.add_argument("--dhs_by_biosample_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--vocabulary_meta_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--biosamples", type=str, nargs="*", help="Biosamples to use")
    parser.add_argument("--center", action="store_true", help="Center the sequences in the summit of the DHS")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--force", action="store_true", help="Force the experiment to run even if data exists")
    args = parser.parse_args()
    return args.vocabulary_path, args.dhs_by_biosample_path, args.vocabulary_meta_path, args.biosamples, args.center, args.interactive, args.force

if __name__ == "__main__":
    vocabulary_path, dhs_by_biosample_path, vocabulary_meta_path, biosamples, center, interactive, force = parse_argument()
    extracter = Extracter(vocabulary_path, dhs_by_biosample_path, vocabulary_meta_path, biosamples, center, interactive, force)
    extracter.run()

