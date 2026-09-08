import os
import urllib.request
import tarfile
import struct
import numpy as np

DATASET_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TAR_PATH = os.path.join(DATA_DIR, "sift.tar.gz")

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    else:
        print(f"File {dest} already exists.")

def extract_tar(tar_path, extract_path):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")

def read_fvecs(filename, max_count=None):
    print(f"Reading {filename}...")
    vectors = []
    with open(filename, "rb") as f:
        count = 0
        while True:
            if max_count is not None and count >= max_count:
                break
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec_bytes = f.read(dim * 4)
            vec = struct.unpack("f" * dim, vec_bytes)
            vectors.append(vec)
            count += 1
            if count % 20000 == 0:
                print(f"  read {count} vectors")
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename, max_count=None):
    print(f"Reading {filename}...")
    vectors = []
    with open(filename, "rb") as f:
        count = 0
        while True:
            if max_count is not None and count >= max_count:
                break
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec_bytes = f.read(dim * 4)
            vec = struct.unpack("i" * dim, vec_bytes)
            vectors.append(vec)
            count += 1
    return np.array(vectors, dtype=np.int32)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(DATASET_URL, TAR_PATH)
    
    # Check if extracted files exist
    base_file = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
    if not os.path.exists(base_file):
        extract_tar(TAR_PATH, DATA_DIR)
        
    print("Dataset ready.")

if __name__ == "__main__":
    main()
