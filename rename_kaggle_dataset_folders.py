from pathlib import Path
import os
import json

data_path = Path("data/dataset")

data = Path(os.path.abspath(os.path.join(data_path, os.pardir)))

with open(data / "cat_to_name.json", "r") as file:
    jfile = json.load(file)
    class_names_dict = {int(k): jfile[k] for k in jfile}

# Compensate for Folder Naming
# class_names_from_dataset = train_data.classes
# class_names = [class_names_dict[int(k)] for k in class_names_from_dataset]
# class_to_idx = {val: i for (i, val) in enumerate(class_names)}

# print(class_names_dict)

directories = os.listdir(data_path)

for dir in directories:
    subdirectories = [subdir for subdir in os.listdir(data_path/ dir) if os.path.isdir(data_path/ dir / subdir)]
    # subdirectories = os.listdir(data_path/ dir)
    print(subdirectories)
    for sub in subdirectories:
        old_path =os.path.abspath(data_path/ dir / sub)
        new_path = os.path.abspath(data_path/ dir / class_names_dict[int(sub)])
        # print(old_path)
        # print(new_path)
        os.rename(old_path, new_path)

# TODO: zip up dataset

# for dirpath, dirnames, filenames in os.walk(data_path):
#     print(Path(dirpath).stem)


