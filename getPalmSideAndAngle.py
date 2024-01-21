import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Путь к файлу с чекпоинтом
checkpoint_file_path = "saved_angle_checkpoint.pth"

# Загрузка состояния модели из файла
loaded_checkpoint = torch.load(checkpoint_file_path)

# Создание модели с соответствующей архитектурой
model_angle = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model_angle.fc.in_features
model_angle.fc = nn.Linear(num_ftrs, 1)

# Загрузка сохраненного состояния в модель
model_angle.load_state_dict(loaded_checkpoint['model_state_dict'])

# Создание оптимизатора с теми же параметрами, что и во время тренировки
optimizer_angle = optim.SGD(model_angle.parameters(), lr=0.001, momentum=0.9)
optimizer_angle.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

# Перевод модели в режим оценки (выключение Dropout и BatchNorm)
model_angle.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(model_angle, model, image_path, transform, label_map):
    # Load the image
    image = Image.open(image_path)

    # Apply the transformations
    image = transform(image)

    # Add an extra dimension for the batch
    image = image.unsqueeze(0)

    # Make sure the image tensor is on the same device as the models
    image = image.to(next(model.parameters()).device)

    # Make the predictions
    output_angle = model_angle(image)
    output_side = model(image)

    # Extract the angle prediction (it comes out as a single-item tensor, so we take the value)
    angle = output_angle.item()

    # Extract the side prediction
    _, side_index = torch.max(output_side, 1)
    side = label_map[side_index.item()]

    return angle, side

label_map = {0: "Right", 1: "Left"}
image_path = "path_to_your_image.png"
angle, side = predict(model_angle, model_angle, image_path, transform, label_map)
print(f"Predicted angle: {angle}")
print(f"Predicted side: {side}")
