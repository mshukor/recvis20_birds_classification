# Some basic setup
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import sys
import argparse

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--input-folder', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--output-folder-crop', type=str, default='crop_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--output-folder-mask', type=str, default='mask_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--generate-masks', type=str, default='mask_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")


args = parser.parse_args()


# Define a Mask-R-CNN model in Detectron2
cfg = get_cfg()
cfg.merge_from_file("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Detection Threshold 
cfg.MODEL.ROI_HEADS.NMS = 0.4 # Non Maximum Suppression Threshold 
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
model = DefaultPredictor(cfg)


def detect_birds(model, input_folder, output_folder_crop, generate_masks=False, output_folder_mask="mask_dataset"):
  kernel = np.ones((43, 43), 'uint8')
  for data_folder in list(os.listdir(input_folder)): # Iterate over train, val and test
    non_cropped = 0
    non_cropped_names = []
    num_imgs = 0
    directory = input_folder+'/'+data_folder
    print("\nDetecting birds on :", data_folder)
    for folder in list(os.listdir(directory)): # Iterate over classes of birds
      size = len(list(os.listdir(directory+'/'+folder)))
      num_imgs += size
      os.makedirs(output_folder_crop, exist_ok = True)
      os.makedirs(output_folder_crop+'/'+data_folder+'/'+folder, exist_ok = True)
      if generate_masks:
        os.makedirs(output_folder_mask, exist_ok = True)
        os.makedirs(output_folder_mask+'/'+data_folder+'/'+folder, exist_ok = True)
        

      img_paths = []          
      img_detections = [] 

      # Reformat weird images
      for file in os.listdir(directory+'/'+folder):
        i = plt.imread(directory+'/'+folder+'/'+file)
        if len(i.shape)==2 or i.shape[2]!=3:
          i = Image.fromarray(i)
          i = i.convert('RGB')
          i.save(directory+'/'+folder+'/'+file)
        del i
                
      # Get image paths and detections : not the most efficient way, but it avoids defining a proper detectron2-specific Dataloader 
      for img_path in list(os.listdir(directory+'/'+folder)):
        img = cv2.imread(directory+'/'+folder+'/'+img_path)
        with torch.no_grad():
          detections = model(img)["instances"]
        img_paths.append(directory+'/'+folder+'/'+img_path)
        img_detections.append(detections)

      # Save cropped images
      for (path, detections) in (zip(img_paths, img_detections)):
        img = np.array(Image.open(path))

        # Bounding boxes and labels of detections
        if len(detections.scores)>0:

          # Get the most probable bird prediction bounding box
          index_birds = np.where(detections.pred_classes.cpu().numpy()==14)[0] # 14 is the default class number for bird
          if len(index_birds)==0:
            # Flip the image if we are not able to detect the bird
            non_cropped_names.append(path)
            non_cropped += 1
            path = path.split("/")[-1]
            plt.imsave(output_folder_crop+'/'+data_folder+'/'+folder+'/'+path, np.array(ImageOps.mirror(Image.fromarray(img))), dpi=1000)
            plt.close()  
            continue
          bird = int(torch.max(detections.scores[index_birds],0)[1].cpu().numpy())
          [x1,y1,x2,y2]=detections.pred_boxes[index_birds][bird].tensor[0].cpu().numpy()
          mask = detections.pred_masks.cpu().numpy().astype(np.uint8).squeeze()
          count=1
          invalid_mask = False


          # If we are able to detect the bird, enlarge the bounding box and generate a new image
          x1, y1 = np.maximum(0,int(x1)-20), np.maximum(0,int(y1)-20)
          x2, y2 = np.minimum(x2+40,img.shape[1]), np.minimum(y2+40,img.shape[0])
          
          # generate mask
          if generate_masks:
            if len(mask.shape) > 2:
              invalid_mask = True
            else:
              imgcv = cv2.imread(path)
              dilate_img = cv2.dilate(mask, kernel, iterations=1)
              masked_img = cv2.bitwise_and(imgcv, imgcv, mask = dilate_img)
            
          img = img[int(np.ceil(y1)):int(y2), int(np.ceil(x1)):int(x2), :]
          # crop the masked image
          if not invalid_mask:
            masked_img = masked_img[int(np.ceil(y1)):int(y2), int(np.ceil(x1)):int(x2), :]

          # Save generated image with detections
          path = path.split("/")[-1]
          plt.imsave(output_folder_crop+'/'+data_folder+'/'+folder+'/crop'+path, img, dpi=1000)
          if generate_masks:
            if not invalid_mask:
              masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
              plt.imsave(output_folder_mask+'/'+data_folder+'/'+folder+'/'+path, masked_img, dpi=1000)
          plt.close()   
          
        else:
          # Flip the image if we are not able to detect the bird
          non_cropped_names.append(path)
          non_cropped+=1
          path = path.split("/")[-1]
          # Flip the image if we are not able to detect it
          plt.imsave(output_folder_crop+'/'+data_folder+'/'+folder+'/crop'+path, np.array(ImageOps.mirror(Image.fromarray(img))), dpi=1000)

          plt.close()  

    print("\t{}% of {} images non cropped".format(np.round(100*non_cropped/num_imgs,2),data_folder))
  return(non_cropped_names)

non_cropped_paths = detect_birds(model=model, input_folder=args.input_folder, output_folder_crop=args.output_folder_crop, output_folder_mask = args.output_folder_mask, generate_masks=args.generate_masks)
