from helper import download_file_if_not_exists
from pycocotools.coco import COCO
import random


url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
directory = "./"
filename = "annotations_trainval2017.zip"

download_file_if_not_exists(url, directory, filename)



train_samples = 10
val_samples = 5

for data_type in ['train', 'val']:
  annFile = 'annotations/instances_{}2017.json'.format(data_type)
  # initialize COCO api for instance annotations
  coco=COCO(annFile)


  #get category ids
  catNms = ['car','bus','truck']
  print("loading category ids for" + str(catNms))
  catIds = coco.getCatIds(catNms=catNms);


  #Get image Ids
  imgIds = set()
  print("loading image ids for" + str(catNms))
  for id in catIds:
    imgIds.update(coco.getImgIds(catIds=id ));
  nsamples = train_samples if data_type == 'train' else val_samples
  imgIds = random.sample(imgIds, nsamples)
  print("Number of images: " + str(len(imgIds)))

  #Get annotations
  print("loading annotations")
  annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
  anns = coco.loadAnns(annIds)
  print("Number of annotations " + str(len(annIds)))

  #Creating annotatoin dictionary
  maping_dic = {3:0,6:0,8:0}
  annotations_dic = {}
  for ann in anns:
    if ann['image_id'] not in annotations_dic:
      annotations_dic[ann['image_id']] = []
    #normalize bbox
    img = coco.loadImgs(ann['image_id'])
    width = img[0]['width']
    height = img[0]['height']
    bbox_width = ann['bbox'][2]
    bbox_height = ann['bbox'][3]
    x_min = ann['bbox'][0]
    y_min = ann['bbox'][1]

    x_center = x_min + bbox_width / 2
    y_center = y_min + bbox_height / 2

    ann['bbox'][0] = x_center / width
    ann['bbox'][1] = y_center / height
    ann['bbox'][2] = bbox_width /  width
    ann['bbox'][3] = bbox_height / height
    #convert to yolo format
    sample = [maping_dic[ann['category_id']]] + ann['bbox']
    annotations_dic[ann['image_id']].append(" ".join(map(str, sample)))
  print("Number of images with annotations: " + str(len(annotations_dic.keys())))

  #Convert coco annotation to yolo
  # Iterate over the dictionary and create files
  import os
  os.makedirs('cocodataset/labels/{}'.format(data_type), exist_ok=True)
  for key, values in annotations_dic.items():
      # Create a file with the name as the dictionary key
      file_name = (12-len(str(key)))*'0' + str(key)
      with open(f'cocodataset/labels/{data_type}/{file_name}.txt', 'w') as file:
          # Write each list element on a separate line
          for value in values:
              file.write(value + '\n')
  #download images
  import requests
  imgs = coco.loadImgs(imgIds)
  os.makedirs('cocodataset/images/{}/'.format(data_type),exist_ok=True)

  for img in imgs:
    url = img['coco_url']
    filename = url.split('/')[-1]
    r = requests.get(url, allow_redirects=True)
    open("cocodataset/images/{}/".format(data_type)+filename, 'wb').write(r.content)
