# README of analytics and data science over this project


## Command to run experiments from project directory
**Please be aware of the paths**
- metadata_path is the path towards the Extracted sequences csv, which is basically the informations about the sequences extracted from the hg38 genome
- data_path is the path towards the one hot encoded sequences that contains a dictionary of:
  - "X": one hot encoding of the character sequences
  - "y": one hot encoding of the components (16 components)
  - "widths": lengths of the sequences
python3 analytics/experiments.py --metadata_path /home/bio/PhD/projects/COURSES/DiffusionSynthBio/results/DHS_extracted_seqs.csv --data_path /home/bio/PhD/projects/COURSES/DiffusionSynthBio/results/DHS_one_hot.npz --center --interactive
