import os
from pathlib import Path
from Bio import SeqIO

from utils.constants import GENOME_FILE_NAME

def download_hg38_genome_or_load():
    # Download the human genome from
    project_dir = Path(__file__).resolve().parents[1]
    tmp_dir = project_dir / "tmp"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Check if the file is already downloaded
    if not os.path.exists(tmp_dir / GENOME_FILE_NAME):
        print(f"Genome is not present. Downloading the genome file {GENOME_FILE_NAME}...")
        # Down and save it to the tmp directory
        os.system(f"wget 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/{GENOME_FILE_NAME}' -O {tmp_dir}/{GENOME_FILE_NAME}")
        os.system(f"gunzip {tmp_dir}/{GENOME_FILE_NAME}")
    else:
        print(f"The genome file {GENOME_FILE_NAME} is already downloaded.")
    
    i = 0

    return SeqIO.parse(tmp_dir / "hg38.fa", "fasta")
