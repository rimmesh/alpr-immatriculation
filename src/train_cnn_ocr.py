import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
DATA_DIR = "data/ocr_chars"
MODEL_OUT = "models/ocr/cnn_ocr.pth"
os.makedirs("models/ocr", exist_ok=True)

# Image transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Class names (ex: 0-9, A-Z)
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Load pretrained CNN (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Adjust last layer for our characters
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train
print("Training CNN...")
for epoch in range(5):  # simple & fast finetuning
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images.repeat(1,3,1,1))  # convert 1-channel → 3-channel
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/5 | Loss = {total_loss:.4f}")

torch.save(model.state_dict(), MODEL_OUT)
print("Model saved →", MODEL_OUT)
