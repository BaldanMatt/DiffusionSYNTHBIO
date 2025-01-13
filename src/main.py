from pathlib import Path

def main(data_dir):
    print("Starting main function...")
    print(f"\tThe data directory is located at: {data_dir}")
    pass


if __name__ == "__main__":
    # retrieve the current working directory and resolve its parent
    parent_dir = Path(__file__).resolve().parent.parent
    
    # init data directory
    data_dir = parent_dir / "data"

    main(data_dir)

