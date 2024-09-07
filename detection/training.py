from ultralytics import YOLO

model = YOLO("yolov10s.pt")  # build a new model from YAML
results = model.train(data="custom_coco.yaml", epochs=50, imgsz=640)

print("Training end, find the best model in runs/")
## If you faced any problems with loading the dataset edit the .yaml file with the absolute path of the dataset ##