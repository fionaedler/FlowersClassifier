from pathlib import Path

import torch.nn
import torchvision
import matplotlib.pyplot as plt

import engine
import model_builder
import prep_data

# device agnostic code!
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up data_path
data_path = Path("data/dataset")

# Download model and adjust to our problem
effnetb0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
effnetb0_transform = effnetb0_weights.transforms()

# create DataLoader
train_dataloader, test_dataloader, class_names = prep_data.create_dataloaders(data_path=data_path, transform=effnetb0_transform)


# Call function to create model and adjust output to number of classes
effnetb0_model = model_builder.create_effnetb0_model(num_classes=len(class_names)).to(device)


# create optimizer and loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=effnetb0_model.parameters(), lr=0.001)

# TODO: call training/testing loops
EPOCHS = 1
# engine.train(model=effnetb0_model,
#              train_dataloader=train_dataloader,
#              test_dataloader=test_dataloader,
#              optimizer=optimizer,
#              loss_fn=loss_fn,
#              epochs=EPOCHS,
#              device=device)


# TODO: save out model


# TODO: create files for predictions
# TODO: plot predictions


# # Inspect Dataloader
# inputs, targets = next(iter(train_dataloader))
# img = inputs[0]
# label = targets[0].item()
# plt.imshow(img.permute(1, 2,0))
# plt.title(f"{label}: {class_names[label]}")
# plt.show()