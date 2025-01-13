docker run --rm -it -v "$(pwd)/data:/app/data" -v "$(pwd)/src:/app/src" sbexam:latest python3 src/load_data.py

