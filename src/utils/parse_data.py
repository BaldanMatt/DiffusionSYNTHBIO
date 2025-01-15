import polars as pl
import numpy as np
from pathlib import Path
from Bio import SeqIO
# We need to import the Bio.io to read fasta files
from utils.download_hg38_genome import download_hg38_genome_or_load

def parse_data(data: pl.DataFrame, metadata: pl.DataFrame):
    print("Parsing data...")
    # We need to download the human genome
    genome_parser = download_hg38_genome_or_load()
    # We need to read the genome based on the content of metadata
    # in metadata i have three columns (seqname, start, end)
    # for each row in metadata we need to extract the sequence from the genome
    fast_dict = SeqIO.to_dict(genome_parser)
    extracted_seq = {}
    for row in metadata.select(["seqname", "start", "end"]).head(10).iter_rows(): 
        seqname, start, end = row
        print(f"Extracting sequence {seqname} from {start} to {end}")
        print(f"Types are: {type(seqname)} {type(start)} {type(end)}")
        if seqname not in fast_dict:
            raise ValueError(f"The sequence {seqname} is not present in the genome.")
        seq_record = fast_dict[seqname]
        subseq = seq_record.seq
        region_name=f"{seqname}:{start}-{end}"
        extracted_seq[region_name]=str(subseq)[start:end]
        print(f"The extracted sequence is: {extracted_seq[region_name]}")
    
    return

if __name__ == "__main__":
    parse_data()

