from pathlib import Path
import requests
import zipfile

# unzip dataset
def download_and_unzip():
    source = "https://github.com/fionaedler/FlowersClassifier/raw/main/data/dataset.zip"
    target_file = Path(source).name
    data_path = Path("FlowersClassifier/data/")

    with open(data_path / target_file, "wb") as f:
        request = requests.get(source)
        print(f"[INFO] Downloading {target_file} from {source}...")
        # print(f"[INFO] Downloading file from {source}...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
        print(f"[INFO] Unzipping {target_file} data...")
        zip_ref.extractall(Path(data_path))
