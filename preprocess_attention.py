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
from torchvision import transforms
from tqdm import tqdm


# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

from models import WSDAN
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--input-folder', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--output-folder', type=str, default='attention_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")


args = parser.parse_args()

model_path = 'experiment/wsdanp_retrain_model_9.pth'
NET = 'inception_mixed_7c' #'"inception_mixed_6e"  #inception_mixed_7c
num_attentions = 32   

model = WSDAN(num_classes=20, M=num_attentions, net=NET, pretrained=False)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint) 
model.cuda()
def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


tf = transforms.ToTensor()
ToPILImage = transforms.ToPILImage()

def detect_birds(model, input_folder, output_folder):
  kernel = np.ones((43, 43), 'uint8')
  for data_folder in list(os.listdir(input_folder)): # Iterate over train, val and test
    # if data_folder != "val_images":
    #   continue
    non_cropped = 0
    non_cropped_names = []
    num_imgs = 0
    directory = input_folder+'/'+data_folder
    print("\nDetecting birds on :", data_folder)
    for folder in list(os.listdir(directory)): # Iterate over classes of birds
      size = len(list(os.listdir(directory+'/'+folder)))
      num_imgs += size
      os.makedirs(output_folder, exist_ok = True)
      os.makedirs(output_folder+'/'+data_folder+'/'+folder, exist_ok = True)
 
      img_paths = []          
      img_detections = [] 

                 
      # Get image paths and detections : not the most efficient way, but it avoids defining a proper detectron2-specific Dataloader 
      for img_path in tqdm(list(os.listdir(directory+'/'+folder))):
        img = plt.imread(directory+'/'+folder+'/'+img_path)
        X = tf(img)
        X  = X.unsqueeze(0)
        X = X.cuda()
        # img = cv2.imread(directory+'/'+folder+'/'+img_path)
        with torch.no_grad():
          _, _, attention_maps = model(X)
        img_paths.append(directory+'/'+folder+'/'+img_path)
        img_detections.append(attention_maps)
        

      # Save cropped images
      for (path, detections) in (zip(img_paths, img_detections)):
        img = plt.imread(path)
        X = tf(img)
        X = X.float()
        attention_maps = F.upsample_bilinear(detections, size=(X.size(1), X.size(2)))
        attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())
        
        img = ToPILImage(X.float()*attention_maps[0][0].cpu())
        # img =  img.float()*attention_maps
 
        # Save generated image with detections
        path = path.split("/")[-1]
        img = np.array(img)
        img = img.astype('uint8')
        plt.imsave(output_folder+'/'+data_folder+'/'+folder+'/'+path, img, dpi=1000)
        plt.close()   

    print("\t{}% of {} images non cropped".format(np.round(100*non_cropped/num_imgs,2),data_folder))
  return(non_cropped_names)

non_cropped_paths = detect_birds(model=model, input_folder=args.input_folder, output_folder=args.output_folder)
