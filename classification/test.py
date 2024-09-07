import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model_cls.pth',map_location=torch.device('cpu')))
#model = torch.load('path_to_model.pth', map_location=torch.device('cpu'))

model = model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),# ResNet18 expects 224x224 input images
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(10),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3952, 0.3817, 0.3761], std=[0.2371, 0.2313, 0.2301]),
])


def predict_image(image_path, model, transform, class_names):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]

class_names = {0:'car',1:'bus',2:'truck'}
image_path = 'img.jpg'
prediction = predict_image(image_path, model, transform, class_names)
print(f'Predicted Class: {prediction}')