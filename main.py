from pathlib import Path

import torch.nn
import torchvision
import matplotlib.pyplot as plt

import engine
import model_builder
import prep_data

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up data_path
data_path = Path("data/dataset")

# Download model and adjust to our problem
effnetb2_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
effnetb2_transform = effnetb2_weights.transforms()

# create DataLoaders
train_dataloader, test_dataloader, class_names = prep_data.create_dataloaders(data_path=data_path, transform=effnetb2_transform)
# print(class_names)

# Call function to create model and adjust output to number of classes
effnetb2_model = model_builder.create_effnetb2_model(num_classes=len(class_names)).to(device)



# create optimizer and loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=effnetb2_model.parameters(), lr=0.001)

# call training/testing loops
# TODO: Save out model parameters every x epochs?

EPOCHS = 1
# effnetb2_results = engine.train(model=effnetb2_model,
#              train_dataloader=train_dataloader,
#              test_dataloader=test_dataloader,
#              optimizer=optimizer,
#              loss_fn=loss_fn,
#              epochs=EPOCHS,
#              device=device)



# save out model
model_dir = "models"
model_name = "effnetb2_flower_classifier.pth"
model_builder.save_model(model=effnetb2_model,
                 target_dir = model_dir,
                 model_name = model_name)


# TODO: create scripts for predictions
# TODO: plot predictions


# # Inspect Dataloader
# inputs, targets = next(iter(train_dataloader))
# img = inputs[0]
# label = targets[0].item()
# plt.imshow(img.permute(1, 2,0))
# plt.title(f"{label}: {class_names[label]}")
# plt.show()