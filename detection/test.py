from ultralytics import YOLO

# Load a model
model = YOLO("yolov10_best.pt")  # pretrained YOLOv8n model
results = model.predict("img.jpg", save_crop = True, conf = 0.5, save = True,show= True)  # return a list of Results objects
print(results)