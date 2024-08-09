import torchvision
from torchvision import datasets, transforms
from pathlib import Path
import json
from torch.utils.data import DataLoader
import os


def create_dataloaders(data_path: Path,
                       transform: torchvision.transforms = None):

    # Setup Paths to dataset
    data_path = data_path
    data = Path(os.path.abspath(os.path.join(data_path, os.pardir)))
    train_dir = data_path / "train"
    test_dir = data_path / "valid"
    valid_dir = data_path / "test"

    # Create Temporary Manual Transforms if no transform is passed in
    if transform is None:
        IMG_SIZE = 224
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            torchvision.transforms.ToTensor()
        ])
    # TODO: add data augentation for train data?

    # Create Dataset and DataLoader
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_dataloader = DataLoader(train_data,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_data,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True)


    # Extract Class names from Json file
    with open(data / "cat_to_name.json", "r") as file:
        jfile = json.load(file)
        class_names_dict = {int(k): jfile[k] for k in jfile}

    # Compensate for Folder Naming
    class_names_from_dataset = train_data.classes
    class_names = [class_names_dict[int(k)] for k in class_names_from_dataset]
    class_to_idx = {val: i for (i, val) in enumerate(class_names)}

    # Inspect Dataloader
    # inputs, targets = next(iter(train_dataloader))
    # img = inputs[0]
    # label = targets[0].item()
    # plt.imshow(img.permute(1, 2,0))
    # plt.title(f"{label}: {class_names[label]}")
    # plt.show()

    return train_dataloader, test_dataloader, class_names
