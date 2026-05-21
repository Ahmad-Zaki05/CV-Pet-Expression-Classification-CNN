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
# Load all data without transforms first to split
full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transforms.ToTensor())
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

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, 
    [train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"\nDataset split:")
print(f"  Training: {len(train_dataset)} samples ({TRAIN_RATIO*100:.0f}%)")
print(f"  Validation: {len(val_dataset)} samples ({VALID_RATIO*100:.0f}%)")
print(f"  Test: {len(test_dataset)} samples ({TEST_RATIO*100:.0f}%)")

# Apply transforms: augmentation for training, no augmentation for val/test
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

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
criterion = nn.CrossEntropyLoss()

# %%

learning_rate = 1e-3  # Higher LR for training from scratch with batch norm
optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=0.9)

# Learning rate scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, 
    min_lr=1e-6
)

num_epochs = 50
train_losses = []
val_losses = []
best_val_accuracy = 0.0
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
criterion = nn.CrossEntropyLoss()


# %%
# Train DenseNet121
learning_rate = 1e-3  # Higher LR for training from scratch
optimizer = torch.optim.Adam(densenet121.parameters(), lr=learning_rate)

# Learning rate scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3,
    min_lr=1e-6
)

num_epochs = 100
train_losses = []
val_losses = []
best_val_accuracy = 0.0
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
# # ResNet50

# %%
class ResidualBlock(nn.Module):
    """
    Bottleneck residual block used in ResNet50/101/152.
    Reduces parameters by using 1x1 -> 3x3 -> 1x1 conv sequence.
    expansion=4 means the output channels are 4x the base width.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # 3x3 conv (stride applied here for spatial downsampling)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 1x1 conv to expand channels back (out_channels * 4)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu  = nn.ReLU(inplace=True)

        # Downsample shortcut: needed when dimensions change so residual
        # connection shapes match the main path output
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))          # No ReLU before adding residual

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                          # Skip connection
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """
    ResNet50: 4 stages with [3, 4, 6, 3] bottleneck blocks.
    Input: (B, 3, 224, 224) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes=4):
        super(ResNet50, self).__init__()

        self.in_channels = 64

        # Stem: initial 7x7 conv + maxpool (224 -> 56)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 residual stages
        # Stage 1: no spatial downsampling (stride=1), 64 base -> 256 out channels
        self.layer1 = self._make_layer(out_channels=64,  num_blocks=3, stride=1)
        # Stage 2: spatial downsampling (stride=2), 128 base -> 512 out channels
        self.layer2 = self._make_layer(out_channels=128, num_blocks=4, stride=2)
        # Stage 3: spatial downsampling (stride=2), 256 base -> 1024 out channels
        self.layer3 = self._make_layer(out_channels=256, num_blocks=6, stride=2)
        # Stage 4: spatial downsampling (stride=2), 512 base -> 2048 out channels
        self.layer4 = self._make_layer(out_channels=512, num_blocks=3, stride=2)

        # Head
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * ResidualBlock.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Builds one residual stage.
        The first block handles any change in spatial size or channel count
        via a downsample shortcut. Remaining blocks keep dimensions unchanged.
        """
        downsample = None

        # Downsample shortcut needed if stride != 1 (spatial change)
        # or if channels don't match (first block of each stage)
        if stride != 1 or self.in_channels != out_channels * ResidualBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * ResidualBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResidualBlock.expansion)
            )

        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * ResidualBlock.expansion

        # Remaining blocks: stride=1, no downsampling needed
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))

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


resnet50 = ResNet50(num_classes=num_classes).to(dv)
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ## Training

# %%
learning_rate = 1e-3
optimizer = torch.optim.SGD(resnet50.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3,
    min_lr=1e-6
)

num_epochs = 50
train_losses = []
val_losses = []
best_val_accuracy = 0.0
best_model_path = "best_resnet50.pth"

print("Starting ResNet50 training...")
for epoch in range(num_epochs):
    # Training phase
    resnet50.train()
    epoch_loss = 0.0

    for images, labels in train_dataloader:
        images, labels = images.to(dv), labels.to(dv)
        outputs = resnet50(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation phase
    resnet50.eval()
    val_epoch_loss = 0.0
    val_preds, val_true = [], []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(dv), labels.to(dv)
            outputs = resnet50(images)
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
        torch.save(resnet50.state_dict(), best_model_path)
        print(f"  → Model saved! (Val Accuracy: {val_accuracy:.4f})")

    scheduler.step(avg_val_loss)

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
    
    # Text-based confusion matrices
    val_cm = confusion_matrix(val_true, val_preds)
    test_cm = confusion_matrix(test_true, test_preds)
    
    print(f"\nValidation Confusion Matrix:")
    import pandas as pd
    val_cm_df = pd.DataFrame(val_cm, index=class_names, columns=[f'Pred_{c}' for c in class_names])
    print(val_cm_df)
    
    print(f"\nTest Confusion Matrix:")
    test_cm_df = pd.DataFrame(test_cm, index=class_names, columns=[f'Pred_{c}' for c in class_names])
    print(test_cm_df)
    
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

# Evaluate the DenseNet121 model
vgg16.load_state_dict(torch.load("./saved/best_vgg16.pth"))
densenet121.load_state_dict(torch.load("./saved/best_densenet121.pth"))
vgg16_results = evaluate_model(vgg16, "VGG16", val_dataloader, test_dataloader, criterion, dv, class_names)
densenet_results = evaluate_model(densenet121, "DenseNet121", val_dataloader, test_dataloader, criterion, dv, class_names)
resnet50.load_state_dict(torch.load("./saved/best_resnet50.pth"))
resnet50_results = evaluate_model(resnet50, "ResNet50", val_dataloader, test_dataloader, criterion, dv, class_names)


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
