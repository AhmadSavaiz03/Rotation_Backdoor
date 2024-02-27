import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
rotation_angle = 45  # Angle for rotation
poisoning_rate = 0.05  # 5% of the dataset to be poisoned
target_label = 0  # Target label for poisoned data

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Number of classes in CIFAR-10
num_classes = 10

def poison_dataset(dataset, rotation_angle, poisoning_rate, target_label):
    poisoned_indices = random.sample(range(len(dataset)), int(len(dataset) * poisoning_rate))
    poisoned_data = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        if i in poisoned_indices:
            img = transforms.functional.rotate(img, rotation_angle)
            label = target_label  # Set to the target label
        poisoned_data.append((img, label))

    return poisoned_data

# Poison the training dataset
train_dataset = poison_dataset(train_dataset, rotation_angle, poisoning_rate, target_label)

# Load and modify the model
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training the model
for epoch in range(10):  # Adjust epochs as needed
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{10}, Loss: {total_loss}')

# Evaluation function
def evaluate_model(model, data_loader, is_poisoned=False, target_label=0):
    correct = 0
    total = 0
    attack_success = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # Corrected line
            if is_poisoned:
                images = torch.stack([transforms.functional.rotate(image, rotation_angle) for image in images.to("cpu")]).to(device)
                labels = torch.full_like(labels, target_label)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if is_poisoned:
                attack_success += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    attack_success_rate = 100 * attack_success / total if is_poisoned else None
    return accuracy, attack_success_rate

# Evaluating Clean Data Accuracy (CDA)
cda, _ = evaluate_model(model, test_loader)
print(f'Clean Data Accuracy: {cda}%')

# Evaluating Attack Success Rate (ASR)
_, asr = evaluate_model(model, test_loader, is_poisoned=True, target_label=target_label)
print(f'Attack Success Rate: {asr}%')