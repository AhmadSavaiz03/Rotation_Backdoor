import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import random
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
rotation_angle = 45  # As used in the paper for one of the experiments
poisoning_rate = 0.05  # 5% poisoning rate as an example
target_label = 0

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load GTSRB (or any other dataset used in the paper)
train_dataset = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)

# Determine the number of unique labels (classes)
unique_labels = set()
for _, label in train_dataset:
    unique_labels.add(label)
num_classes = len(unique_labels)

print(f'Train Dataset size: {len(train_dataset)}')
print(f'Test Dataset size: {len(test_dataset)}')

def poison_dataset(dataset, rotation_angle, poisoning_rate, target_label=0):
    poisoned_indices = random.sample(range(len(dataset)), int(len(dataset) * poisoning_rate))
    poisoned_data = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        if i in poisoned_indices:
            img = transforms.functional.rotate(img, rotation_angle)
            label = target_label  # Target label as defined in the paper
        poisoned_data.append((img, label))

    return poisoned_data

train_dataset = poison_dataset(train_dataset, rotation_angle, poisoning_rate)

# Load and modify the model
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(10):  # Number of epochs can be adjusted based on the paper
    total_loss = 0
    print(f'Processing Epoch {epoch+1}/{10}')
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f'Epoch {epoch+1}/{10}, Loss: {total_loss}')

def evaluate_model(model, data_loader, is_poisoned=False, target_label=0):
    correct = 0
    total = 0
    attack_success = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if is_poisoned:
                images = torch.stack([transforms.functional.rotate(image, rotation_angle) for image in images])
                labels = torch.full_like(labels, target_label)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if is_poisoned:
                attack_success += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    attack_success_rate = 100 * attack_success / total if is_poisoned else None
    print(f'Batch Processed, Correct Predictions: {correct}, Total: {total}')
    return accuracy, attack_success_rate

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
poisoned_test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)  # Use part of the training dataset as poisoned dataset for testing

# Evaluating Clean Data Accuracy (CDA)
cda, _ = evaluate_model(model, test_loader, is_poisoned=False, target_label=target_label)
print(f'Clean Data Accuracy: {cda}%')

# Evaluating Attack Success Rate (ASR)
_, asr = evaluate_model(model, poisoned_test_loader, is_poisoned=True, target_label=target_label)
print(f'Attack Success Rate: {asr}%')