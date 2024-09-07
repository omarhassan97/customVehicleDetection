from ultralytics import YOLO
import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


### Loading models ####
##predictoin model
print("start the program")
model = YOLO("detection/yolov10_best.pt")  # pretrained YOLOv8n model

## Classification model
model_cls = models.resnet18(weights=False)
model_cls.fc = nn.Linear(512, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_cls.load_state_dict(torch.load('classification/model_cls.pth',map_location=torch.device('cpu')))
#model_cls.load_state_dict(torch.load('classification/model_cls.pth')) #use this line with GPU 

#model = torch.load('path_to_model.pth', map_location=torch.device('cpu'))

model_cls = model_cls.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),# ResNet18 expects 224x224 input images
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(10),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3952, 0.3817, 0.3761], std=[0.2371, 0.2313, 0.2301]),
])

# This function detect vehicle and classify the detected vehicle.
# Function input: image_path
# Function output: annotated_image_path, dictionary of predicted classes {'car': 2, 'bus':1, 'truck',0}
# Load a model
def detectClassify(image_path):

	##Detection
	#Remove prevous run files
	if os.path.exists("runs"):
		shutil.rmtree("runs")

	results = model.predict(image_path, save_crop = True, conf = 0.5, save = True)  # return a list of Results objects
	base_name = os.path.basename(image_path)
	annotated_image = os.path.join("runs\detect\predict", base_name)

	#get paths of croped images to classify each
	folder_path = "runs\detect\predict\crops"

	# List to store file paths
	file_paths = []

	# Walk through the directory
	for root, directories, files in os.walk(folder_path):
	    for file_name in files:
	        # Join root and file_name to get the full path
	        file_path = os.path.join(root, file_name)
	        file_paths.append(file_path)

	#print(file_paths)



	#classification
	def predict_image(image_path, model, transform, class_names):
	    model_cls.eval()
	    image = Image.open(image_path)
	    image = transform(image).unsqueeze(0)  # Add batch dimension
	    image = image.to(device)
	    
	    with torch.no_grad():
	        output = model(image)
	        _, predicted = torch.max(output, 1)
	    
	    return class_names[predicted.item()]

	class_names = {0:'bus',1:'car',2:'truck'}
	predictions = {"bus":0,"car":0,'truck':0}
	for image_path in file_paths:
		class_name = predict_image(image_path, model_cls, transform, class_names)
		predictions[class_name] += 1
	print(f'Predicted Class: {predictions}')
	return annotated_image,predictions

# image_path = "img.jpg"
# print("annotated image_path " + str(detectClassify(image_path)))