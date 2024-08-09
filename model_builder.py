import torch
import torchvision

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