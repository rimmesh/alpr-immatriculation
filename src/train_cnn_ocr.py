import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = "data/ocr_chars"
MODEL_OUT = "models/ocr/cnn_ocr_multilang.pth"
os.makedirs("models/ocr", exist_ok=True)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

classes = dataset.classes
print("Classes detected:", classes)
num_classes = len(classes)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("\nTraining multilingual OCR CNN...\n")
for epoch in range(10):
    total = 0
    for imgs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(imgs.repeat(1,3,1,1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += loss.item()

    print(f"Epoch {epoch+1}/10 | Loss: {total:.4f}")

torch.save(model.state_dict(), MODEL_OUT)
print("\nModel saved to:", MODEL_OUT)
