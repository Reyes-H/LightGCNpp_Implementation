import os
import zipfile
import urllib.request

def download_movielens_1m(save_path="data/ml-1m"):
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "ml-1m.zip"

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("Extraction complete.")

if __name__ == "__main__":
    download_movielens_1m()
