import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights

class PalmDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file, sep=";")
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"Right": 0, "Left": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = self.img_dir + "/" + img_name
        image = Image.open(img_path)
        label = self.label_map[self.df.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Add this line
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


trainset = PalmDataset(csv_file="data/train/label.csv", img_dir="data/train", transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = PalmDataset(csv_file="data/test/label.csv", img_dir="data/test", transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10  # Define the number of epochs

# Training loop
for epoch in range(num_epochs):
    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))