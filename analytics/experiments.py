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
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

class Extracter:
    RANDOM_STATE = 42

    PROJ_DIR = str(pathlib.Path(__file__).parent.parent)
    RES_DIR = PROJ_DIR + "/" + 'results'
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
            res_dhs_by_biosample_one_hot_name = self.RES_DIR + "/" + 'DHS_one_hot_samples'
            res_dhs_by_biosample_meta_name = self.RES_DIR + "/" + 'DHS_sequences_samples'
            for biosample in self.biosamples:
                # Build file name concatenating the biosamples
                res_dhs_by_biosample_one_hot_name = res_dhs_by_biosample_one_hot_name + "_{}".format(biosample)
                res_dhs_by_biosample_meta_name = res_dhs_by_biosample_meta_name + "_{}".format(biosample)

            self.res_dhs_by_biosample_one_hot_name = res_dhs_by_biosample_one_hot_name
            self.res_dhs_by_biosample_meta_name = res_dhs_by_biosample_meta_name
        else:
            self.res_dhs_by_biosample_one_hot_name = "DHS_one_hot"
            self.res_dhs_by_biosample_meta_name = "DHS_sequences"
        if self.center:
            self.res_dhs_by_biosample_meta_name += "_centered"
            self.res_dhs_by_biosample_one_hot_name += "_centered"
        self.res_dhs_by_biosample_meta_name += ".csv"
        self.res_dhs_by_biosample_one_hot_name += ".npz"
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
        metadata = metadata.reset_index(drop=True)
        # metadata, data = eda_validate_and_clean(metadata, data)
        # Summarize metadata
        # exploratory_data_analysis(metadata, data, interactive)
        # Extract relative positions
        metadata["relative_start"] = 0
        metadata["relative_end"] = metadata["end"] - metadata["start"]
        metadata["relative_summit"] = metadata["summit"] - metadata["start"]

        # #Extract distance to closest DSS
        # import HTSeq
        # import itertools
        # gtffile = HTSeq.GFF_Reader(
        #     "/home/bio/PhD/projects/COURSES/DiffusionSynthBio/data/Homo_sapiens.GRCh37.56_chrom1.gtf")
        # # gtffile = HTSeq.GFF_Reader("/home/bio/PhD/projects/COURSES/DiffusionSynthBio/data/GCF_000001405.39_GRCh38.p13_genomic_subsample.gff.gz")
        # tsspos = set()
        # for feature in gtffile:
        #     if feature.type == "exon" and feature.attr["exon_number"] == "1":
        #         tsspos.add(feature.iv.start_d_as_pos)
        # for i, row in tqdm(metadata.iterrows()):
        #     chr = row["seqname"].strip("chr")
        #     # Use dictionary lookup for TSS positions by chromosome
        #     if not hasattr(self, 'tss_by_chrom'):
        #         # Create chromosome dictionary on first use
        #         self.tss_by_chrom = {}
        #         for pos in tsspos:
        #             self.tss_by_chrom.setdefault(pos.chrom, []).append(pos.pos)
        #
        #     # Get TSS positions for this chromosome
        #     chr_tss_positions = self.tss_by_chrom.get(chr, [])
        #     if chr_tss_positions:
        #         # Calculate minimum distance directly
        #         metadata.at[i, "closest_tss_distance"] = min(abs(pos - row["start"]) for pos in chr_tss_positions)
        #     else:
        #         metadata.at[i, "closest_tss_distance"] = None
        # plotter.plot_hist_distance_to_tss(metadata, xlabel="Distance to closest TSS (bp)", ylabel="#DHS", scale="log")

        # Extract metrics
        # matches = mt.find_subsequence(data["X"], "TATA", "ACGTN")
        # plotter.plot_find_subsequence(matches, metadata, hue="component")

        # Find GC islands
        motifs = ["GC"*i for i in range(1,4)]
        for seq in motifs:
            matches = mt.find_subsequence(data["X"], seq, "ACGTN")
            # Count how many sequences have at least one GC island
            num_sequences_with_gc_islands = (matches.sum(axis=1)!=0).sum()
            print("Ratio of sequences with GC islands: {:.2f}%".format(num_sequences_with_gc_islands / data["X"].shape[0] * 100))
            plotter.plot_find_subsequence(matches, metadata, hue="component")
        matches = mt.find_subsequence(data["X"], "GGGCG", "ACGTN")
        # Count how many sequences have at least one GC island
        num_sequences_with_gc_islands = (matches.sum(axis=1) != 0).sum()
        print("Ratio of sequences with GC islands: {:.2f}%".format(
            num_sequences_with_gc_islands / data["X"].shape[0] * 100))
        plotter.plot_find_subsequence(matches, metadata, hue="component")
        matches = mt.find_subsequence(data["X"], "TATAAA", "ACGTN")
        # Count how many sequences have at least one GC island
        num_sequences_with_gc_islands = (matches.sum(axis=1) != 0).sum()
        print("Ratio of sequences with TATA box: {:.2f}%".format(
            num_sequences_with_gc_islands / data["X"].shape[0] * 100))
        plotter.plot_find_subsequence(matches, metadata, hue="component")
        matches = mt.find_subsequence(data["X"], "CCAAAT", "ACGTN")
        # Count how many sequences have at least one GC island
        num_sequences_with_gc_islands = (matches.sum(axis=1) != 0).sum()
        print("Ratio of sequences with CCAAT-box: {:.2f}%".format(
            num_sequences_with_gc_islands / data["X"].shape[0] * 100))
        plotter.plot_find_subsequence(matches, metadata, hue="component")
        matches = mt.find_subsequence(data["X"], "ACTTCAC", "ACGTN")
        # Count how many sequences have at least one GC island
        num_sequences_with_gc_islands = (matches.sum(axis=1) != 0).sum()
        print("Ratio of sequences with CCAAT-box: {:.2f}%".format(
            num_sequences_with_gc_islands / data["X"].shape[0] * 100))
        plotter.plot_find_subsequence(matches, metadata, hue="component")

        # # Reducer
        # X_flat = data["X"].reshape(data["X"].shape[0], -1)
        # reducer = umap.UMAP(random_state = self.RANDOM_STATE)
        # X_tsne = TSNE(n_components=2, learning_rate="auto", perplexity=3, random_state=self.RANDOM_STATE).fit_transform(X_flat)
        # X_umap = reducer.fit_transform(X_flat)
        # df = pd.DataFrame(np.hstack([X_umap,X_tsne]), columns=["UMAP_1", "UMAP_2","TSNE_1","TSNE_2"])
        # df = pd.concat([df, metadata.reset_index(drop=True)], axis=1)
        # plotter.plot_umap(df, x=["UMAP_1","TSNE_1"], y=["UMAP_2","TSNE_2"], hue="closest_tss_distance", palette="magma")

        # Extract diversity
        div = mt.diversity(data["X"])
        print('Diversity: ', div)
        metadata = metadata.reset_index(drop=True)
        for comp in metadata["component"].unique():
            comp_indexes = metadata[metadata["component"] == comp].index
            comp_data = data["X"][comp_indexes,:,:]
            comp_div = mt.diversity(comp_data)
            print('Diversity of component {}: '.format(comp), comp_div)

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
        # Create data already save the vocabulary res file
        X = du.create_data(vocabulary, center_in_summit=self.center, output_file=self.res_dhs_by_biosample_meta_name)
        one_hot_x, one_hot_y, widths = du.parse_data(X)
        print("Saving one-hot encoded data... (sequences)")

        # Saving the extracted one coded sequences
        # remove the last part after the last / in seqspath
        np.savez_compressed(
            self.res_dhs_by_biosample_one_hot_name,
            X=one_hot_x,
            y=one_hot_y,
            widths=widths
        )
        return one_hot_x, one_hot_y, widths

class Plotter:
    def __init__(self,
                 style: str = "whitegrid",
                 figsize: tuple = (10,6),
                 show: bool = True,):
        sns.set_style(style)
        self.figsize=figsize
        self.show=show

    def _finalize_plot(self,fig):
        if self.show:
            plt.show()
        else:
            if self.save_path is not None:
                fig.savefig(self.save_path)
            else:
                print("Save path is Not provided!")

    def plot_hist_distance_to_tss(self, metadata, title='', xlabel='', ylabel='', save_path = None, scale:str="log"):
        fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=self.figsize)
        if scale == "log":
            sns.histplot(data=metadata, x="closest_tss_distance", hue = "component", log_scale=True, ax=ax)
        else:
            sns.histplot(data=metadata, x="closest_tss_distance", hue="component", ax=ax)
        ax.set_xlim(0, 1e6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self._finalize_plot(fig)

    def plot_find_subsequence(self, matches, metadata, hue=None, title='', xlabel='', ylabel='', save_path=None):
        def add_summit_lines(ax, data):
            ax.axvline(x=data["summit_25"].iloc[0], color="red", linestyle="--", linewidth=1)
            ax.axvline(x=data["summit_50"].iloc[0], color="red", linestyle="-", linewidth=1)
            ax.axvline(x=data["summit_75"].iloc[0], color="red", linestyle="--", linewidth=1)

        def set_axis_labels(ax):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        if hue is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=self.figsize)
            frequencies = matches.sum(axis=0)
            sns.lineplot(data=frequencies, ax=ax)
            add_summit_lines(ax, metadata)
            set_axis_labels(ax)
        else:
            grouped_matches = []
            for value in metadata[hue].unique():
                mask = metadata[hue] == value
                group_frequencies = matches[mask].sum(axis=0)
                summit_stats = metadata.loc[mask, "relative_summit"]
                grouped_matches.append(pd.DataFrame({
                    'position': range(len(group_frequencies)),
                    'frequency': group_frequencies,
                    'summit_25': np.percentile(summit_stats, 25),
                    'summit_50': np.percentile(summit_stats, 50),
                    'summit_75': np.percentile(summit_stats, 75),
                    'hue': value
                }))
            grouped_df = pd.concat(grouped_matches)
            grid = sns.relplot(data=grouped_df, x='position', y='frequency', col='hue', kind='line',
                               height=3, aspect=1.5, col_wrap=4,
                               facet_kws=dict(sharex=False, sharey=False))
            for ax, comp in zip(grid.axes.flat, metadata[hue].unique()):
                add_summit_lines(ax, grouped_df[grouped_df['hue'] == comp])
                set_axis_labels(ax)
                ax.set_title(f'Component {comp}')
                median_summit = grouped_df[grouped_df['hue'] == comp]['summit_50'].iloc[0]
                ax.text(median_summit, ax.get_ylim()[1] * 0.9,
                        f'Summit: {int(median_summit)}',
                        horizontalalignment='center')
            fig = grid.figure

        self._finalize_plot(fig)

    def plot_umap(self, umap_df, x: str | list = "", y: str | list = "", palette="", hue = None, title='', xlabel='', ylabel='', save_path = None):
        n_plots = len(x)
        if len(y) != n_plots:
            raise ValueError("Number of y values and number of x values are not equal")
        fig, axs = plt.subplots(ncols = n_plots, figsize=self.figsize)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            sns.scatterplot(umap_df, x=x[i], y=y[i], hue=hue, palette=palette, ax=ax, s=10, legend=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        self._finalize_plot(fig)

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

