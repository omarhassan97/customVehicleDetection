# Vehicle Detection and Classification using Custom YOLOv10s Model and ResNet18
This project focuses on vehicle detection and classification using a custom YOLOv10s model. The model is trained to detect vehicles in images and classify them into different categories. The project includes dataset preparation, model training, testing, and evaluation, and offers a simple web page for model inference using FastAPI.

<p align="center">
  <img src="https://github.com/omarhassan97/customVehicleDetection/blob/main/static/output.png" alt="example" />
</p>
## Project Structure
* Object Detection Model: YOLOv10s custom model for vehicle detection
* Vehicle Classification: Classify detected vehicles into specific categories

## Setup Instructions
1. install [Anaconda](https://www.anaconda.com/)
2. Create a new environment and activate it
```
conda create -n vehicle_detection python=3.9
conda activate vehicle_detection
```
   
3. Install required dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

## Dataset Preparation

Use the provided `datasetDownloader.py` script to download and prepare the datasets for both detection and classification tasks:

#### For vehicle detection:
```
python detection/datasetDownloader.py

```

#### for vehicle classification:
```
python classification/datasetDownloader.py
```


This script will download the datasets and create the required annotations for training the models. The script transform the annotatoin from coco to yolo annotatoin style.


## Model Testing

 

#### For vehicle detection:
```
python detection/test.py

```

#### for vehicle classification:
Download model weights from the link [model_cls.pth](https://drive.google.com/file/d/1Mio_Cli-jBPtciy4-TenivEheWdT-qG-/view?usp=sharing) and place it in folder `../classification`

```
python classification/test.py
```

## Model Traning
 This step 
#### For vehicle detection:
```
python detection/test.py

```

#### for vehicle classification:
```
python classification/test.py
```

##  FastAPI application

To run the fastapi application for easy testing do the following:
1. Download classificatoin model weights from the link [model_cls.pth](https://drive.google.com/file/d/1Mio_Cli-jBPtciy4-TenivEheWdT-qG-/view?usp=sharing) and place it in folder `../classification`
2. use `uvicorn main:app --reload` in anacona prompt to run the app
3. in your browser use the link `http://127.0.0.1:8000/` to open the app

