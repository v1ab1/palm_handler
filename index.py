import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

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
        # Add angle value
        angle = self.df.iloc[idx, 1]  
        label = self.label_map[self.df.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)
        return image, label, angle  # Return image, label, and angle

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

num_epochs = 10

trainset = PalmDataset(csv_file="data/train/label.csv", img_dir="data/train", transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = PalmDataset(csv_file="data/test/label.csv", img_dir="data/test", transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Create a model for angle regression
model_angle = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model_angle.fc.in_features
model_angle.fc = nn.Linear(num_ftrs, 1)

criterion_mse = nn.MSELoss()  # or nn.L1Loss() for MAE
optimizer_angle = torch.optim.SGD(model_angle.parameters(), lr=0.001, momentum=0.9)

# Training loop for angle regression
for epoch in range(num_epochs):
    for images, labels, angles in tqdm(trainloader):
        outputs = model_angle(images)
        loss = criterion_mse(outputs.view(-1), angles.float())  # MSE loss
        loss.backward()
        optimizer_angle.step()
        optimizer_angle.zero_grad()

# Testing loop for angle regression
total_error = 0
total = 0
with torch.no_grad():
    for images, labels, angles in tqdm(testloader):
        outputs = model_angle(images)
        error = torch.abs(outputs.view(-1) - angles.float())  # Absolute error
        total_error += error.sum().item()
        total += angles.size(0)

mae = total_error / total
print(f'Accuracy: {mae}')

checkpoint = {
    'model_state_dict': model_angle.state_dict(),
    'optimizer_state_dict': optimizer_angle.state_dict(),
}

# Путь, куда сохранить полное состояние
checkpoint_file_path = "saved_angle_checkpoint.pth"

# Сохранение состояния модели и оптимизатора
torch.save(checkpoint, checkpoint_file_path)
print(f"Состояние модели и оптимизатора сохранено в файл: {checkpoint_file_path}")
