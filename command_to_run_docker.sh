docker run --rm -it -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/src:/app/src" \
    sbexam:latest \
    python3 src/scratch_parse.py \
        --data_dir_path "/app/data" \
        --data_file_name "test_all_classifier_light.csv.gz" \
        --metadata_file_name "DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
