import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

# Пути к файлам с чекпоинтами для обеих моделей
checkpoint_file_angle_path = "saved_angle_checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_checkpoint_angle = torch.load(checkpoint_file_angle_path, map_location=device)

model_angle = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model_angle.fc.in_features
model_angle.fc = nn.Linear(num_ftrs, 1)

model_angle.load_state_dict(loaded_checkpoint_angle['model_state_dict'])

model_angle.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(model_angle, model_side, image_path, transform, label_map):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    output_angle = model_angle(image).item()

    output_side_probabilities = model_side(image)
    _, predicted_side_index = torch.max(output_side_probabilities, 1)
    side = label_map[predicted_side_index.item()]

    return output_angle, side

while (True):
    label_map = {0: "Правая", 1: "Левая"}
    image_path = input('Введите путь к картинке')
    angle, side = predict(model_angle, model_angle, image_path, transform, label_map)
    print(f"Угол: {angle}")
    print(f"Сторона: {side}")
