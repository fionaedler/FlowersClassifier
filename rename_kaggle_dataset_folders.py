from pathlib import Path
import os
import json
import shutil

dataset_path = Path("data/dataset")
data = Path("data")

# Get dict to convert numbers into class names
with open(data / "cat_to_name.json", "r") as file:
    jfile = json.load(file)
    class_names_dict = {int(k): jfile[k] for k in jfile}

# traverse through directories and rename to class names
try:
    directories = os.listdir(dataset_path)
    for dir in directories:
        subdirectories = [subdir for subdir in os.listdir(dataset_path / dir) if os.path.isdir(dataset_path / dir / subdir)]
        for sub in subdirectories:
            old_path =os.path.abspath(dataset_path / dir / sub)
            new_path = os.path.abspath(dataset_path / dir / class_names_dict[int(sub)])
            os.rename(old_path, new_path)
except:
    print("Directories don't have the correct format to be converted to class names. Maybe they have been converted already?")

# zip up dataset
shutil.make_archive("data/dataset", 'zip', dataset_path)



