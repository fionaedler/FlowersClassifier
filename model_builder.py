import torch
import torchvision
from pathlib import Path

def create_effnetb2_model(num_classes: int):
    # Get pretrained weights and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Update Classifier Head to fit our output
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

        Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
          either ".pth" or ".pt" as the file extension.

        Example usage:
        save_model(model=model_0,
                   target_dir="models",
                   model_name="effnetb2_model.pth")
        """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

def load_model(
            # model: torch.nn.Module,
               model_path: str,
               num_classes: int,
               device: torch.device = "cpu"
               ):

    model = create_effnetb2_model(num_classes)
    model.load_state_dict(torch.load(f=model_path, map_location=torch.device(device))    )
    return model