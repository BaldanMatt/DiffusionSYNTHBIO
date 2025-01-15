import polars as pl

# Define the name of the genome file
GENOME_FILE_NAME = "hg38.fa.gz"

# Define the schema for the metadata files
DHS_metadata_schema = {"seqname": pl.String(),
                       "start": pl.Int32(),
                       "end": pl.Int32(),
                       "identifier": pl.String(),
                       "mean_signal": pl.Float32(),
                       "numsamples": pl.Int32(),
                       "summit": pl.Int32(),
                       "core_start": pl.Int32(),
                       "core_end": pl.Int32(),
                       "component": pl.String(),}

BIOSAMPLES_metadata_schema = {"library_order": pl.Int32(),
                              "Biosample name": pl.String(),
                              "Vocabulary representation": pl.String(),
                              "DCC Experiment ID": pl.String(),
                              "DCC Library ID": pl.String(),
                              "DCC Biosample ID": pl.String(),
                              "DCC File ID": pl.String(),
                              "Altius Aggregation ID": pl.String(),
                              "Altius Library ID": pl.String(),
                              "Altius Biosample ID": pl.String(),
                              "Replicate indicators": pl.String(),
                              "System": pl.String(),
                              "Subsystem": pl.String(),
                              "Organ": pl.String(),
                              "Biosample type": pl.String(),
                              "Biological state": pl.String(),
                              "Germ layer": pl.String(),
                              "Description": pl.String(),
                              "Growth stage": pl.String(),
                              "Age": pl.String(),
                              "Sex": pl.String(),
                              "Ethnicity": pl.String(),
                              "Donor ID": pl.String(),
                              "Unique cellular conditions": pl.Int32(),
                              "Used in Figure 1b": pl.Int8(),
                              "Biosample protocol": pl.String(),
                              "Experiment protocol": pl.String(),
                              "Library kit method": pl.String(),
                              "Library cleanup": pl.String(),
                              "DNase I units/mL": pl.Float32(),
                              "Amount Nucleic Acid (ng)": pl.String(), # This is a string because it is not always a number (there are some occurrences of <2ng or too low and such...)
                              "Nuclei count": pl.Int32(),
                              "Protease inhibitor": pl.String(),
                              "Library sequencing date": pl.Date(),
                              "Reads used": pl.Int32(),
                              "DCC SPOT score": pl.Float32(),
                              "Per-biosample peaks": pl.Int32(),
                              "DHSs in Index": pl.Int32(),}


