# README of analytics and data science over this project


## Command to run experiments from project directory
**Please be aware of the paths**
- metadata_path is the path towards the Extracted sequences csv, which is basically the informations about the sequences extracted from the hg38 genome
- data_path is the path towards the one hot encoded sequences that contains a dictionary of:
  - "X": one hot encoding of the character sequences
  - "y": one hot encoding of the components (16 components)
  - "widths": lengths of the sequences
python3 analytics/experiments.py --vocabulary_path /home/bio/PhD/projects/COURSES/DiffusionSynthBio/data/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz --dhs_by_biosample_path /home/bio/PhD/projects/COURSES/DiffusionSynthBio/data/dat_bin_FDR01_hg38.mtx.gz --vocabulary_meta_path /home/bio/PhD/projects/COURSES/DiffusionSynthBio/data/DHS_Index_and_Vocabulary_metadata.tsv --biosamples GM12878 K562 HepG2 --center --interactive --force