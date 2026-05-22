# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
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
import seaborn as sns
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

DATASET_DIR = "./pets-facial-expression-dataset"

assert os.path.exists(DATASET_DIR), f"Dataset directory '{DATASET_DIR}' does not exist."

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

#hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 0 # Adjust based on your system's capabilities


# %% [markdown]
# # Prepare Datasets

# %%
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandAugment(num_ops=10, magnitude=3),  # Apply BEFORE ToTensor (operates on PIL images)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# %% [markdown]
# # Load Datasets

# %%
# Custom Subset class to avoid transform leakage
class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform if transform is not None else transforms.ToTensor()
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        # Always apply transform - guaranteed to return tensor
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.indices)

# Load all data WITHOUT transforms (need PIL images for RandAugment)
full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=None)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")
print(f"Total samples: {len(full_dataset)}")

# Create 70/15/15 split
total_size = len(full_dataset)
train_size = int(total_size * TRAIN_RATIO)
valid_size = int(total_size * VALID_RATIO)
test_size = total_size - train_size - valid_size

# Create indices for random split
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(total_size, generator=generator).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + valid_size]
test_indices = indices[train_size + valid_size:]

# Use CustomSubset to apply transforms independently (NO LEAKAGE)
train_dataset = CustomSubset(full_dataset, train_indices, transform=train_transform)
val_dataset = CustomSubset(full_dataset, val_indices, transform=val_test_transform)
test_dataset = CustomSubset(full_dataset, test_indices, transform=val_test_transform)

print(f"\nDataset split:")
print(f"  Training: {len(train_dataset)} samples ({TRAIN_RATIO*100:.0f}%)")
print(f"  Validation: {len(val_dataset)} samples ({VALID_RATIO*100:.0f}%)")
print(f"  Test: {len(test_dataset)} samples ({TEST_RATIO*100:.0f}%)")


# %% [markdown]
# # Create DataLoaders

# %%
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

print("\n DataLoaders created successfully!")

# count samples per class in training dataset
def count_samples_per_class(dataset, class_names):
    class_counts = {class_name: 0 for class_name in class_names}
    for _, label in dataset:
        class_counts[class_names[label]] += 1
    return class_counts

train_class_counts = count_samples_per_class(train_dataset, class_names)
print("\nTraining dataset class distribution:")
for class_name, count in train_class_counts.items():
    print(f"{class_name}: {count} samples")

val_class_counts = count_samples_per_class(val_dataset, class_names)
print("\nValidation dataset class distribution:")
for class_name, count in val_class_counts.items():
    print(f"{class_name}: {count} samples")

test_class_counts = count_samples_per_class(test_dataset, class_names)
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
# # Dataset Analysis & Diagnostics

# %%
# Verify data loading and class distribution
print("="*60)
print("DATASET DIAGNOSTICS")
print("="*60)

print("\n1. Dataset Sizes:")
print(f"   Total images: {total_size}")
print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

print("\n2. Training Set Statistics:")
print(f"   Total training samples: {len(train_dataset)}")
for cls, count in train_class_counts.items():
    pct = (count / len(train_dataset)) * 100
    print(f"     {cls}: {count:3d} samples ({pct:5.1f}%)")

print("\n3. Sample shapes and value ranges:")
sample_batch = next(iter(train_dataloader))
images, labels = sample_batch
print(f"   Batch shape: {images.shape}")
print(f"   Min pixel value: {images.min():.4f}")
print(f"   Max pixel value: {images.max():.4f}")
print(f"   Mean pixel value: {images.mean():.4f}")
print(f"   Std pixel value: {images.std():.4f}")

print("\n4. Label distribution in batch:")
for i, label in enumerate(labels[:8]):
    print(f"     Image {i}: {class_names[label]}")

print("\n5. Training iterations per epoch:")
print(f"   Train batches: {len(train_dataloader)}")
print(f"   Val batches: {len(val_dataloader)}")
print(f"   Test batches: {len(test_dataloader)}")
print("\n" + "="*60)


# %% [markdown]
# # VGG16

# %%
class VGG16(nn.Module): # vgg16 configuration C 
    def __init__(self, num_classes=4):
        super(VGG16,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
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
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
vgg16 = VGG16(num_classes=num_classes).to(dv)
if os.path.exists("best_vgg16.pth"):
    vgg16.load_state_dict(torch.load("best_vgg16.pth"))
    print("Loaded best_vgg16.pth")
else:
    print("best_vgg16.pth not found, training from scratch")
criterion = nn.CrossEntropyLoss()

# %%

learning_rate = 1e-3  # Higher LR for training from scratch with batch norm
optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=0.9)

# Learning rate scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, 
    min_lr=1e-6
)

# Evaluate loaded model to establish baseline
vgg16.eval()
baseline_preds, baseline_true = [], []
with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        baseline_preds.extend(predicted.cpu().numpy())
        baseline_true.extend(labels.cpu().numpy())
best_val_accuracy = accuracy_score(baseline_true, baseline_preds)
print(f"Loaded model baseline validation accuracy: {best_val_accuracy:.4f}\n")

num_epochs = 50
train_losses = []
val_losses = []
best_model_path = "best_vgg16.pth"

print("Starting VGG16 training with variable learning rate scheduling...")
for epoch in range(num_epochs):
    # Training phase
    vgg16.train()
    epoch_loss = 0.0
    
    for images, labels in train_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    vgg16.eval()
    val_epoch_loss = 0.0
    val_preds, val_true = [], []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(dv), labels.to(dv)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    avg_val_loss = val_epoch_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    
    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_true, val_preds)
    
    # Save model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(vgg16.state_dict(), best_model_path)
        print(f"  → Model saved! (Val Accuracy: {val_accuracy:.4f})")
    
    # Learning rate scheduler step
    scheduler.step(avg_val_loss)
    
    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.6f}")
    
print("Training complete!")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")


# %% [markdown]
# # DenseNet121

# %%
class Bottleneck(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 4*growth_rate, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self,x):
        out = self.conv1(nn.functional.relu(self.bn1(x)))
        out = self.conv2(nn.functional.relu(self.bn2(out)))
        out = torch.cat([x,out], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_bottlenecks, in_channel, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_bottlenecks):
            layers.append(Bottleneck(in_channel + i*growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self,x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        out = self.conv(nn.functional.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet121(nn.Module):
    def __init__(self, num_blocks, growth_rate=32, num_classes=4):
        super(DenseNet121, self).__init__()
        self.growth_rate = growth_rate

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 2*growth_rate, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(2*growth_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build dense blocks and transitions
        num_channels = 2 * growth_rate
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_block in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(num_block, num_channels, growth_rate))
            num_channels += num_block * growth_rate
            
            # Don't add transition after last dense block
            if i < len(num_blocks) - 1:
                self.transitions.append(Transition(num_channels, num_channels // 2))
                num_channels //= 2

        # Final layers
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        
        # Dense blocks and transitions
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Final layers
        x = nn.functional.relu(self.bn_final(x))
        x = self.pool_final(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
densenet121 = DenseNet121(num_blocks=[6, 12, 24, 16], growth_rate=32, num_classes=num_classes).to(dv)
if os.path.exists("best_densenet121.pth"):
    densenet121.load_state_dict(torch.load("best_densenet121.pth"))
    print("Loaded best_densenet121.pth")
else:
    print("best_densenet121.pth not found, training from scratch")
criterion = nn.CrossEntropyLoss()


# %%
# Train DenseNet121
learning_rate = 1e-3
optimizer = torch.optim.SGD(densenet121.parameters(), lr=learning_rate, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, 
    min_lr=1e-6
)

# Evaluate loaded model to establish baseline
densenet121.eval()
baseline_preds, baseline_true = [], []
with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = densenet121(images)
        _, predicted = torch.max(outputs.data, 1)
        baseline_preds.extend(predicted.cpu().numpy())
        baseline_true.extend(labels.cpu().numpy())
best_val_accuracy = accuracy_score(baseline_true, baseline_preds)
print(f"Loaded model baseline validation accuracy: {best_val_accuracy:.4f}\n")

num_epochs = 50
train_losses = []
val_losses = []
best_model_path = "best_densenet121.pth"

print("Starting DenseNet121 training with variable learning rate scheduling...")
for epoch in range(num_epochs):
    # Training phase
    densenet121.train()
    epoch_loss = 0.0
    
    for images, labels in train_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = densenet121(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    densenet121.eval()
    val_epoch_loss = 0.0
    val_preds, val_true = [], []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(dv), labels.to(dv)
            outputs = densenet121(images)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    avg_val_loss = val_epoch_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    
    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_true, val_preds)
    
    # Save model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(densenet121.state_dict(), best_model_path)
        print(f"  → Model saved! (Val Accuracy: {val_accuracy:.4f})")
    
    # Learning rate scheduler step
    scheduler.step(avg_val_loss)
    
    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.6f}")

print("Training complete!")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")


# %% [markdown]
# # ResNet18

# %%
class BasicBlock(nn.Module):
    """
    Basic residual block used in ResNet18/34.
    Two 3x3 convs with a skip connection. No bottleneck.
    expansion=1 because output channels == input channels (no expansion).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # First 3x3 conv (stride applied here for spatial downsampling)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # Second 3x3 conv (always stride=1, no further downsampling)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))         # No ReLU before skip addition

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                          # Skip connection
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet18: 4 stages with [2, 2, 2, 2] basic blocks.
    Input: (B, 3, 224, 224) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        # Stem: 7x7 conv + maxpool (224 -> 56)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 residual stages — [2,2,2,2] blocks, channels double each stage
        self.layer1 = self._make_layer(out_channels=64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=2, stride=2)

        # Head
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        # expansion=1 so final feature vector is 512, not 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),   # Light dropout: helps on small datasets
            nn.Linear(512 * BasicBlock.expansion, num_classes)
        )

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


resnet18 = ResNet18(num_classes=num_classes).to(dv)
if os.path.exists("best_resnet18.pth"):
    resnet18.load_state_dict(torch.load("best_resnet18.pth"))
    print("Loaded best_resnet18.pth")
else:
    print("best_resnet18.pth not found, training from scratch")
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ## Training

# %%
learning_rate = 1e-2
optimizer = torch.optim.SGD(
    resnet18.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# Evaluate loaded model to establish baseline
resnet18.eval()
baseline_preds, baseline_true = [], []
with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        baseline_preds.extend(predicted.cpu().numpy())
        baseline_true.extend(labels.cpu().numpy())
best_val_accuracy = accuracy_score(baseline_true, baseline_preds)
print(f"Loaded model baseline validation accuracy: {best_val_accuracy:.4f}\n")

num_epochs = 100
train_losses = []
val_losses = []
best_model_path = "best_resnet18.pth"

print("Starting ResNet18 training...")
for epoch in range(num_epochs):
    resnet18.train()
    epoch_loss = 0.0

    for images, labels in train_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = resnet18(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet18.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    resnet18.eval()
    val_epoch_loss = 0.0
    val_preds, val_true = [], []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(dv), labels.to(dv)
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    avg_val_loss = val_epoch_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    val_accuracy = accuracy_score(val_true, val_preds)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(resnet18.state_dict(), best_model_path)
        print(f"  → Model saved! (Val Accuracy: {val_accuracy:.4f})")

    scheduler.step()

    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.6f}")

print("Training complete!")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")


# %% [markdown]
# # MobileNet

# %%
# Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        hidden_dim = in_channels * expand_ratio

        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # No activation here — linear bottleneck by design
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Reduced expansion factor (4 instead of 6) for small dataset:
        # less intermediate width means fewer parameters the small
        # dataset has to supervise in each depthwise conv
        config = [
            # t, c, n, s
            [1, 16,  1, 1],
            [4, 24,  2, 2],   # expansion 6->4
            [4, 32,  3, 2],   # expansion 6->4
            [4, 64,  4, 2],   # expansion 6->4
            [4, 96,  3, 1],   # expansion 6->4
            [4, 160, 3, 2],   # expansion 6->4
            [4, 320, 1, 1],   # expansion 6->4
        ]

        layers = []
        in_channels = 32

        for t, c, n, s in config:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, c, stride, expand_ratio=t))
                in_channels = c

        self.blocks = nn.Sequential(*layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Added Dropout before classifier, consistent with ResNet18
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, num_classes)
        )

        # Kaiming initialization, consistent with ResNet18 and InceptionV3
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


mobilenetv2 = MobileNetV2(num_classes=num_classes).to(dv)

if os.path.exists("best_mobilenetv2.pth"):
    mobilenetv2.load_state_dict(torch.load("best_mobilenetv2.pth"))
    print("Loaded best_mobilenetv2.pth")
else:
    print("best_mobilenetv2.pth not found, training from scratch")

criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ## Training

# %%
learning_rate = 1e-2

optimizer = torch.optim.SGD(
    mobilenetv2.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# Evaluate loaded model to establish baseline
mobilenetv2.eval()

baseline_preds = []
baseline_true = []

with torch.no_grad():

    for images, labels in val_dataloader:

        images = images.to(dv)
        labels = labels.to(dv)

        outputs = mobilenetv2(images)

        _, predicted = torch.max(outputs.data, 1)

        baseline_preds.extend(predicted.cpu().numpy())
        baseline_true.extend(labels.cpu().numpy())

best_val_accuracy = accuracy_score(
    baseline_true,
    baseline_preds
)

print(
    f"Loaded model baseline validation accuracy: "
    f"{best_val_accuracy:.4f}\n"
)

num_epochs = 100

train_losses = []
val_losses = []

best_model_path = "best_mobilenetv2.pth"

print("Starting MobileNetV2 training...")

for epoch in range(num_epochs):

    # Training
    mobilenetv2.train()

    epoch_loss = 0.0

    for images, labels in train_dataloader:

        images = images.to(dv)
        labels = labels.to(dv)

        outputs = mobilenetv2(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            mobilenetv2.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)

    train_losses.append(avg_train_loss)

    # Validation
    mobilenetv2.eval()

    val_epoch_loss = 0.0

    val_preds = []
    val_true = []

    with torch.no_grad():

        for images, labels in val_dataloader:

            images = images.to(dv)
            labels = labels.to(dv)

            outputs = mobilenetv2(images)

            loss = criterion(outputs, labels)

            val_epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    avg_val_loss = val_epoch_loss / len(val_dataloader)

    val_losses.append(avg_val_loss)

    val_accuracy = accuracy_score(
        val_true,
        val_preds
    )

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            mobilenetv2.state_dict(),
            best_model_path
        )

        print(
            f"  → Model saved! "
            f"(Val Accuracy: {val_accuracy:.4f})"
        )

    # Update scheduler
    scheduler.step()

    # Logging
    if (epoch + 1) % 5 == 0:

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"LR: {current_lr:.6f}"
        )

print("Training complete!")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)


# %% [markdown]
# # InceptionV3

# %%
class BasicConv2d(nn.Module):
    """
    Reusable Conv -> BN -> ReLU building block used throughout Inception V3.
    Almost every convolution in the network uses this exact pattern,
    so factoring it out keeps the code clean and consistent.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        # bias=False because BN already handles the bias term
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionA(nn.Module):
    """
    Inception module used in the early stages of the network.
    Four parallel branches are concatenated along the channel dimension:
      - Branch 1: single 1x1 conv
      - Branch 2: 1x1 conv -> 5x5 conv
      - Branch 3: 1x1 conv -> two 3x3 convs (cheaper than one 5x5)
      - Branch 4: avg pool -> 1x1 conv
    The pool_features parameter controls branch 4 output size and varies
    across the three InceptionA modules in the network.
    """
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()

        # Branch 1: simple 1x1
        self.branch1 = BasicConv2d(in_channels, 64, kernel_size=1)

        # Branch 2: 1x1 -> 5x5
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )

        # Branch 3: 1x1 -> 3x3 -> 3x3  (factorized 5x5)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        # Branch 4: avg pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate all branches along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionB(nn.Module):
    """
    Grid reduction module: reduces spatial dimensions from 35x35 -> 17x17.
    Two branches (no pooling branch), then a max pool branch:
      - Branch 1: 1x1 -> 3x3 with stride=2 (spatial downsampling)
      - Branch 2: 1x1 -> 3x3 -> 3x3 with stride=2
      - Branch 3: max pool stride=2
    Stride=2 in branches 1 and 2 (instead of a separate pooling layer)
    is how Inception reduces spatial size without losing information.
    """
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()

        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64,  kernel_size=1),
            BasicConv2d(64,          96,  kernel_size=3, padding=1),
            BasicConv2d(96,          96,  kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)


class InceptionC(nn.Module):
    """
    Factorized convolution module used in the middle stages (17x17 grid).
    The key idea: an nxn convolution is factorized into a 1xn followed by
    an nx1 convolution. This gives the same receptive field at lower cost.
    For example, one 7x7 = 49 multiplications, but 1x7 + 7x1 = 14.
    channels_7x7 controls the intermediate channel count and increases
    across the four InceptionC modules (128 -> 160 -> 160 -> 192).
    """
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        c7 = channels_7x7

        # Branch 1: 1x1
        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=1)

        # Branch 2: 1x1 -> 1x7 -> 7x1
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, c7,  kernel_size=1),
            BasicConv2d(c7,          c7,  kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7,          192, kernel_size=(7, 1), padding=(3, 0))
        )

        # Branch 3: 1x1 -> 7x1 -> 1x7 -> 7x1 -> 1x7  (deeper factorization)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, c7,  kernel_size=1),
            BasicConv2d(c7,          c7,  kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7,          c7,  kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7,          c7,  kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7,          192, kernel_size=(1, 7), padding=(0, 3))
        )

        # Branch 4: avg pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionD(nn.Module):
    """
    Second grid reduction module: 17x17 -> 8x8.
    Same philosophy as InceptionB — uses strided convolutions
    across branches rather than a separate downsampling layer.
    """
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192,         320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192,         192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192,         192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192,         192, kernel_size=3,      stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)


class InceptionE(nn.Module):
    """
    Expanded filter bank module used in the final stages (8x8 grid).
    Branch 2 and 3 each split into two parallel 1x3 and 3x1 convolutions
    whose outputs are concatenated. This maximizes the variety of feature
    detectors at the coarsest spatial scale before classification.
    """
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        # Branch 1: 1x1
        self.branch1 = BasicConv2d(in_channels, 320, kernel_size=1)

        # Branch 2: 1x1 -> split into 1x3 and 3x1 in parallel
        self.branch2_reduce = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 3: 1x1 -> 3x3 -> split into 1x3 and 3x1 in parallel
        self.branch3_reduce = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3_conv   = BasicConv2d(448,          384, kernel_size=3, padding=1)
        self.branch3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 4: avg pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)

        b2 = self.branch2_reduce(x)
        b2 = torch.cat([self.branch2a(b2), self.branch2b(b2)], dim=1)  # 768 ch

        b3 = self.branch3_conv(self.branch3_reduce(x))
        b3 = torch.cat([self.branch3a(b3), self.branch3b(b3)], dim=1)  # 768 ch

        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)  # 320+768+768+192 = 2048 ch


class InceptionV3(nn.Module):
    """
    Inception V3: full architecture as described in
    'Rethinking the Inception Architecture for Computer Vision' (Szegedy et al. 2016).

    Spatial flow:
      Input 224x224 -> Stem -> 26x26 -> InceptionA x3 -> InceptionB
      -> 12x12 -> InceptionC x4 -> InceptionD -> 5x5 -> InceptionE x2
      -> AvgPool -> classifier

    Note: the original paper uses 299x299 input. We use 224x224 to stay
    consistent with the rest of the codebase. The spatial dimensions above
    reflect 224x224 input accordingly.
    """
    def __init__(self, num_classes=4):
        super(InceptionV3, self).__init__()

        # --- Stem ---
        # Aggressively reduces 224x224 down before the Inception modules begin.
        # Uses strided convs rather than pooling to preserve more information.
        self.stem = nn.Sequential(
            BasicConv2d(3,   32,  kernel_size=3, stride=2),           # 224->111
            BasicConv2d(32,  32,  kernel_size=3),                      # 111->109
            BasicConv2d(32,  64,  kernel_size=3, padding=1),           # 109->109
            nn.MaxPool2d(kernel_size=3, stride=2),                     # 109->54
            BasicConv2d(64,  80,  kernel_size=1),
            BasicConv2d(80,  192, kernel_size=3),                      # 54->52
            nn.MaxPool2d(kernel_size=3, stride=2)                      # 52->26
        )

        # --- InceptionA x3 (on 26x26 grid) ---
        # pool_features varies: controls how many channels the avg pool
        # branch contributes; increases to maintain total output channels
        # as the network deepens (256 -> 288 -> 288)
        self.inceptionA1 = InceptionA(192, pool_features=32)   # out: 256ch
        self.inceptionA2 = InceptionA(256, pool_features=64)   # out: 288ch
        self.inceptionA3 = InceptionA(288, pool_features=64)   # out: 288ch

        # --- InceptionB: 26x26 -> 12x12 ---
        self.inceptionB = InceptionB(288)                       # out: 768ch

        # --- InceptionC x4 (on 12x12 grid) ---
        # channels_7x7 increases from 128 to 192 across the four modules,
        # giving the factorized branches progressively more capacity
        self.inceptionC1 = InceptionC(768, channels_7x7=128)   # out: 768ch
        self.inceptionC2 = InceptionC(768, channels_7x7=160)   # out: 768ch
        self.inceptionC3 = InceptionC(768, channels_7x7=160)   # out: 768ch
        self.inceptionC4 = InceptionC(768, channels_7x7=192)   # out: 768ch

        # --- InceptionD: 12x12 -> 5x5 ---
        self.inceptionD = InceptionD(768)                       # out: 1280ch

        # --- InceptionE x2 (on 5x5 grid) ---
        self.inceptionE1 = InceptionE(1280)                     # out: 2048ch
        self.inceptionE2 = InceptionE(2048)                     # out: 2048ch

        # --- Head ---
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(p=0.4)   # Inception V3 uses dropout before classifier
        self.classifier = nn.Linear(2048, num_classes)

        # Kaiming initialization for conv layers, same as ResNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)

        x = self.inceptionB(x)

        x = self.inceptionC1(x)
        x = self.inceptionC2(x)
        x = self.inceptionC3(x)
        x = self.inceptionC4(x)

        x = self.inceptionD(x)

        x = self.inceptionE1(x)
        x = self.inceptionE2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


inceptionv3 = InceptionV3(num_classes=num_classes).to(dv)
if os.path.exists("best_inceptionv3.pth"):
    inceptionv3.load_state_dict(torch.load("best_inceptionv3.pth"))
    print("Loaded best_inceptionv3.pth")
else:
    print("best_inceptionv3.pth not found, training from scratch")
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ## Training

# %%
learning_rate = 1e-2
optimizer = torch.optim.SGD(
    inceptionv3.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# Evaluate loaded model to establish baseline
inceptionv3.eval()
baseline_preds, baseline_true = [], []
with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = inceptionv3(images)
        _, predicted = torch.max(outputs.data, 1)
        baseline_preds.extend(predicted.cpu().numpy())
        baseline_true.extend(labels.cpu().numpy())
best_val_accuracy = accuracy_score(baseline_true, baseline_preds)
print(f"Loaded model baseline validation accuracy: {best_val_accuracy:.4f}\n")

num_epochs = 100
train_losses = []
val_losses = []
best_model_path = "best_inceptionv3.pth"

print("Starting Inception V3 training...")
for epoch in range(num_epochs):
    inceptionv3.train()
    epoch_loss = 0.0

    for images, labels in train_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = inceptionv3(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inceptionv3.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    inceptionv3.eval()
    val_epoch_loss = 0.0
    val_preds, val_true = [], []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(dv), labels.to(dv)
            outputs = inceptionv3(images)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    avg_val_loss = val_epoch_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    val_accuracy = accuracy_score(val_true, val_preds)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(inceptionv3.state_dict(), best_model_path)
        print(f"  → Model saved! (Val Accuracy: {val_accuracy:.4f})")

    scheduler.step()

    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.6f}")

print("Training complete!")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")


# %% [markdown]
# # Evaluation

# %%
# Network-agnostic evaluation function
def evaluate_model(model, model_name, val_loader, test_loader, criterion, device, class_names):
    """
    Evaluate a model on validation and test datasets.
    
    Args:
        model: PyTorch model to evaluate
        model_name: Name of the model for display
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to run on (cpu/cuda)
        class_names: List of class names
    """
    model.eval()
    val_preds, val_true = [], []
    test_preds, test_true = [], []
    val_loss, test_loss = 0.0, 0.0

    # Validation evaluation
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
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
    val_loss /= len(val_loader)

    # Test evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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
    test_loss /= len(test_loader)

    # Print Results
    print("\n" + "="*60)
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nVALIDATION:")
    print(f"  Accuracy:  {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1-Score:  {val_f1:.4f}")
    print(f"  Loss:      {val_loss:.4f}")
    
    print(f"\nTEST:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  Loss:      {test_loss:.4f}")
    
    print(f"\nCONFUSION MATRICES (TEXT):")
    
    val_cm = confusion_matrix(val_true, val_preds)
    test_cm = confusion_matrix(test_true, test_preds)
    
    print(f"\nCONFUSION MATRICES (HEATMAPS):")
    
    # Create figure with two subplots for confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation confusion matrix
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{model_name} - Validation Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Test confusion matrix
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    axes[1].set_title(f'{model_name} - Test Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_loss': val_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_loss': test_loss
    }

# Evaluate the models
if os.path.exists("best_vgg16.pth"):
    vgg16.load_state_dict(torch.load("best_vgg16.pth"))
    vgg16_results = evaluate_model(vgg16, "VGG16", val_dataloader, test_dataloader, criterion, dv, class_names)
if os.path.exists("best_densenet121.pth"):
    densenet121.load_state_dict(torch.load("best_densenet121.pth"))
    densenet_results = evaluate_model(densenet121, "DenseNet121", val_dataloader, test_dataloader, criterion, dv, class_names)
if os.path.exists("best_resnet18.pth"):
    resnet18.load_state_dict(torch.load("best_resnet18.pth"))
    resnet18_results = evaluate_model(resnet18, "ResNet18", val_dataloader, test_dataloader, criterion, dv, class_names)
if os.path.exists("best_mobilenetv2.pth"):
    mobilenetv2.load_state_dict(torch.load("best_mobilenetv2.pth"))
    mobilenetv2_results = evaluate_model(mobilenetv2, "MobileNetV2", val_dataloader, test_dataloader, criterion, dv, class_names)
if os.path.exists("best_inceptionv3.pth"):
    inceptionv3.load_state_dict(torch.load("best_inceptionv3.pth"))
    inceptionv3_results = evaluate_model(inceptionv3, "InceptionV3", val_dataloader, test_dataloader, criterion, dv, class_names)


# %%
# GPU Memory Cleanup
def clear_gpu_memory():
    """
    Clear GPU memory after model evaluation.
    Useful for preventing out-of-memory errors when evaluating multiple models.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory cleared. Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("CUDA not available. No GPU memory to clear.")

# Clear GPU memory after evaluation
clear_gpu_memory()
