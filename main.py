# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: cv_venv (3.14.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup and imports

# %%
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

SEED = 17

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {dv}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATASET_DIR = "./pets-facial-expression-dataset/Master Folder"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

assert os.path.exists(DATASET_DIR), f"Dataset directory '{DATASET_DIR}' does not exist."
assert os.path.exists(TRAIN_DIR), f"Train directory '{TRAIN_DIR}' does not exist."
assert os.path.exists(VALID_DIR), f"Validation directory '{VALID_DIR}' does not exist."
assert os.path.exists(TEST_DIR), f"Test directory '{TEST_DIR}' does not exist."

#hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 4 # Adjust based on your system's capabilities


# %% [markdown]
# # Prepare Datasets

# %%
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Using ImageNet stats for normalization
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Using ImageNet stats for normalization
])

# %% [markdown]
# # Load Datasets

# %%
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VALID_DIR, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_test_transform)
class_names = train_dataset.classes
print(f"Classes: {class_names}")
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
assert train_dataset.classes == val_dataset.classes == test_dataset.classes, "Class labels do not match across datasets."
assert num_classes == 4, f"Expected 4 classes, but found {num_classes}."
assert len(train_dataset) > 0, "Training dataset is empty."
assert len(val_dataset) > 0, "Validation dataset is empty."
assert len(test_dataset) > 0, "Test dataset is empty."

# %% [markdown]
# # Create DataLoaders

# %%
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

print("\n DataLoaders created successfully!")

# count samples per class in training dataset
def count_samples_per_class(dataset):
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
    return class_counts

train_class_counts = count_samples_per_class(train_dataset)
print("\nTraining dataset class distribution:")
for class_name, count in train_class_counts.items():
    print(f"{class_name}: {count} samples")

val_class_counts = count_samples_per_class(val_dataset)
print("\nValidation dataset class distribution:")
for class_name, count in val_class_counts.items():
    print(f"{class_name}: {count} samples")

test_class_counts = count_samples_per_class(test_dataset)
print("\nTest dataset class distribution:")
for class_name, count in test_class_counts.items():
    print(f"{class_name}: {count} samples")


# %% [markdown]
# # View Samples

# %%
def denormalise(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor.cpu()  # Ensure the tensor is on CPU for denormalization
    image_tensor = image_tensor * std + mean  # Denormalize
    image_tensor = torch.clamp(image_tensor, 0, 1)  # Clamp values to [0, 1] range
    return image_tensor

def show_batch(dataloader, class_names, num_images=8):
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    images = [denormalise(img) for img in images]
    
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for plotting
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nShowing a batch of training images:")
show_batch(train_dataloader, class_names)


# %% [markdown]
# # VGG16

# %%
class VGG16(nn.Module): # vgg16 configuration C 
    def __init__(self, num_classes=4):
        super(VGG16,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# %%
# Train VGG16 with monitoring
model = VGG16(num_classes=num_classes).to(dv)
learning_rate = 1e-4  # Increased learning rate
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5
train_losses = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for images, labels in train_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete!")
print(f"Final training loss: {train_losses[-1]:.4f}")

# %%
# Evaluate on Validation and Test Data
model.eval()
val_preds, val_true = [], []
test_preds, test_true = [], []
val_loss, test_loss = 0.0, 0.0

# Validation evaluation
with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        val_preds.extend(predicted.cpu().numpy())
        val_true.extend(labels.cpu().numpy())

val_accuracy = accuracy_score(val_true, val_preds)
val_precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
val_recall = recall_score(val_true, val_preds, average='weighted', zero_division=0)
val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
val_loss /= len(val_dataloader)

print("\nVALIDATION RESULTS")
print(f"Accuracy:  {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1-Score:  {val_f1:.4f}")
print(f"Loss:      {val_loss:.4f}")

# Test evaluation
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_true, test_preds)
test_precision = precision_score(test_true, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(test_true, test_preds, average='weighted', zero_division=0)
test_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
test_loss /= len(test_dataloader)

print("\nTEST RESULTS")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"Loss:      {test_loss:.4f}")

# Confusion Matrices
print("\nCONFUSION MATRICES")
print("\nValidation Confusion Matrix:")
print(confusion_matrix(val_true, val_preds))
print("\nTest Confusion Matrix:")
print(confusion_matrix(test_true, test_preds))
