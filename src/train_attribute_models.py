'''Attribute Classifier Training Module for SignalGuard.

This script trains two attribute classifiers using MobileNetV2:
- Vehicle Type (3 classes: 2-wheeler, 3-wheeler, 4-wheeler)
- Vehicle Color (7 classes: red, blue, black, white, silver, green, yellow)

If real training data is not found, it falls back to a synthetic dataset generated with torchvision's FakeData.
The backbone is frozen; only the final classification head is trained.
Models are saved to `models/type_classifier.pt` and `models/color_classifier.pt`.
''' 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData, ImageFolder
from torchvision.transforms import ToTensor, Resize

# Paths and constants
DATA_ROOT = "data/attributes"
MODEL_DIR = "models"
DEVICE = torch.device("cpu")  # Use CPU by default; use GPU in Colab for faster training.

TYPE_CLASSES = ["2-wheeler", "3-wheeler", "4-wheeler"]
COLOR_CLASSES = ["red", "blue", "black", "white", "silver", "green", "yellow"]

EPOCHS = 15
BATCH_SIZE = 16

def get_dataset(split: str, num_classes: int):
    """Load dataset for the given split (e.g., 'type' or 'color').
    Expected folder layout: DATA_ROOT/<split>/class_name/*.jpg
    If not found or empty, generate synthetic data via FakeData.
    """
    split_path = os.path.join(DATA_ROOT, split)
    if os.path.isdir(split_path) and any(os.scandir(split_path)):
        transform = transforms.Compose([Resize((224, 224)), transforms.ToTensor()])
        return ImageFolder(root=split_path, transform=transform)
    else:
        print(f"Info: {split_path} not found or empty. Using synthetic data for '{split}'.")
        return FakeData(size=10 * num_classes, image_size=(3, 224, 224), num_classes=num_classes, transform=ToTensor())

def train_head(num_classes: int, dataset, model_path: str, description: str):
    print(f"\n--- Training {description} ---")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=1e-3)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved {description} model to {model_path}\n")

def train_attribute_models():
    # Vehicle Type classifier
    type_dataset = get_dataset(split="type", num_classes=len(TYPE_CLASSES))
    train_head(
        num_classes=len(TYPE_CLASSES),
        dataset=type_dataset,
        model_path=os.path.join(MODEL_DIR, "type_classifier.pt"),
        description="Vehicle Type (2/3/4-wheeler)"
    )
    # Vehicle Color classifier
    color_dataset = get_dataset(split="color", num_classes=len(COLOR_CLASSES))
    train_head(
        num_classes=len(COLOR_CLASSES),
        dataset=color_dataset,
        model_path=os.path.join(MODEL_DIR, "color_classifier.pt"),
        description="Vehicle Color (7 classes)"
    )
    print("PHASE 4 COMPLETE")

if __name__ == "__main__":
    train_attribute_models()
