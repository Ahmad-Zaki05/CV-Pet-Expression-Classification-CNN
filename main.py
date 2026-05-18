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
BATCH_SIZE = 64
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
