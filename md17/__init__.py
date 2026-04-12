import os
import urllib.request

def download_dataset_md17(target_dir="./data/MD17"):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://zenodo.org/record/257509/files/md17.tar.gz"
    file_path = os.path.join(target_dir, "md17.tar.gz")
    print(f"Downloading MD17 dataset from {url} ...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Saved to {file_path}. Please extract manually if needed.")