import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from inception import *
from torchvision import datasets

from model import Net
import torch.nn as nn
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
import torchvision

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np

MULTI_SCALE = True
if MULTI_SCALE:
  # Define a Mask-R-CNN model in Detectron2
  cfg = get_cfg()
  cfg.merge_from_file("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Detection Threshold 
  cfg.MODEL.ROI_HEADS.NMS = 0.4 # Non Maximum Suppression Threshold 
  cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
  model_detecron = DefaultPredictor(cfg)


parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--data-crop', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--data-attention', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")                    
parser.add_argument('--data-mask', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")                    

parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--model2', type=str, metavar='M', default=None,
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--model1', type=str, metavar='M', default=None,
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

if use_cuda:
  state_dict = torch.load(args.model)
  if args.model1:
    state_dict1 = torch.load(args.model1)
  # state_dict2 = torch.load(args.model2)
else:
  state_dict = torch.load(args.model, map_location=torch.device('cpu'))

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
# model2 = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)


# model = inception_v3(pretrained=False)
# model.fc = nn.Linear(2048, 20)
# if args.model2:
#   model2 = torchvision.models.resnext50_32x4d(pretrained=True)
#   model2.fc = nn.Linear(2048, 20)
#   state_dict2 = torch.load(args.model2)
#   model2.load_state_dict(state_dict2)
#   model2.eval()
#   model2.cuda()

model.load_state_dict(state_dict)
model.eval()
if args.model1:
  model1 = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
  model1.load_state_dict(state_dict1)
  model1.eval()
  model1.cuda()
# model2.load_state_dict(state_dict2)
# model2.eval()

if use_cuda:
    print('Using GPU')
    model.cuda()
    
    # model2.cuda()
else:
    print('Using CPU')

from data import data_transforms_val

test_dir = args.data + '/val_images'
test_dir_crop = args.data_crop + '/val_images/'
test_dir_attention = args.data_attention + '/val_images'
test_dir_mask = args.data_mask + '/val_images'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

data_transforms = data_transforms_val

DATA = args.data
VALID_IMAGES = '/val_images' #

data_val = datasets.ImageFolder(DATA+ VALID_IMAGES,
                      transform=data_transforms_val)

val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=2, shuffle=False, num_workers=1)

classes_to_names = {v: k for k, v in data_val.class_to_idx.items()}


correct, correct_mask, correct_crop, correct_ensemble, correct_attention = 0, 0, 0, 0, 0
len_data = 0

from torchvision import transforms
data_transforms_boost = transforms.Compose([
    # transforms.Resize((456, 456)),
    # transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
   

    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
])

to_tensor = transforms.ToTensor()
scale1 = [-10, -10]

scale2 = [40, 60]

for d in tqdm(os.listdir(test_dir)):
    for f in tqdm(os.listdir(os.path.join(test_dir, d))):
      if 'jpg' in f:
          len_data+=1
          img = pil_loader(os.path.join(test_dir, d) + '/' + f)
          data = data_transforms(img)
          # data = data.view(1, data.size(0), data.size(1), data.size(2))

          try:
            data_mask = data_transforms(pil_loader(os.path.join(test_dir_mask, d) + '/' + f))
            # data_mask = data_mask.view(1, data.size(0), data.size(1), data.size(2))
          except FileNotFoundError:
            data_mask = data
          try:
            data_crop = data_transforms(pil_loader(os.path.join(test_dir_crop, d) + '/' + f))
          # data_crop = data_crop.view(1, data.size(0), data.size(1), data.size(2))
          except FileNotFoundError:
            data_crop = data

          data_attention = data_transforms(pil_loader(os.path.join(test_dir_attention, d) + '/' + f))
          # data_attention = data_attention.view(1, data.size(0), data.size(1), data.size(2))

          # sample = torch.cat((data_mask, data_mask, data_attention), dim=0)
          # sample.unsqueeze_(0)
          if use_cuda:
              data = data.cuda()
              data_mask = data_mask.cuda()
              data_crop = data_crop.cuda()
              data_attention = data_attention.cuda()
          data = data.unsqueeze(0)
          data_mask = data_mask.unsqueeze(0)
          data_crop = data_crop.unsqueeze(0)
          data_attention = data_attention.unsqueeze(0)


          output = model(data)
          pred_score, pred = output.data.max(1, keepdim=False)
          cls = classes_to_names[int(pred.item())].split(".")[1]
          if cls.lower() in f.lower():
            correct+=1
          if MULTI_SCALE:
            img = np.asarray(img)
            with torch.no_grad():
              detections = model_detecron(img)["instances"]
            index_birds = np.where(detections.pred_classes.cpu().numpy()==14)[0] # 14 is the default class number for bird
            if len(index_birds) > 0:
              bird = int(torch.max(detections.scores[index_birds],0)[1].cpu().numpy())
              [x1,y1,x2,y2]=detections.pred_boxes[index_birds][bird].tensor[0].cpu().numpy()
              x11, y11 = np.maximum(0,int(x1)-scale1[0]), np.maximum(0,int(y1)-scale1[0])
              x12, y12 = np.minimum(x2+scale1[1],img.shape[1]), np.minimum(y2+scale1[1],img.shape[0])

              img_scale_1 = img[int(np.ceil(y11)):int(y12), int(np.ceil(x11)):int(x12), :]
              img_scale_1 = data_transforms(Image.fromarray(img_scale_1)).unsqueeze(0).cuda()

              x21, y21 = np.maximum(0,int(x1)-scale2[0]), np.maximum(0,int(y1)-scale2[0])
              x22, y22 = np.minimum(x2+scale2[1],img.shape[1]), np.minimum(y2+scale2[1],img.shape[0])

              img_scale_2 = img[int(np.ceil(y21)):int(y22), int(np.ceil(x21)):int(x22), :]
              img_scale_2 = data_transforms(Image.fromarray(img_scale_2)).unsqueeze(0).cuda()

            else:
              img_scale_2 = data
              img_scale_1 = data
            print(img_scale_1.shape)
            # output_mask = model(data_mask)
          # pred_mask_score, pred_mask = output_mask.data.max(1, keepdim=False)
          # cls_mask = classes_to_names[int(pred_mask.item())].split(".")[1]
          # if cls_mask.lower() in f.lower():
          #   correct_mask+=1
          # if args.model2:
          #   output_mask = model2(data)
          #   pred_mask_score, pred_mask = output_mask.data.max(1, keepdim=False)
          #   cls_mask = classes_to_names[int(pred_mask.item())].split(".")[1]
          #   if cls_mask.lower() in f.lower():
          #     correct_mask+=1



          # output_crop = model(data_crop)
          output_crop = model(data_crop)
          pred_crop_score, pred_crop = output_crop.data.max(1, keepdim=False)
          cls_crop = classes_to_names[int(pred_crop.item())].split(".")[1]
          if cls_crop.lower() in f.lower():
            correct_crop+=1

          # output_aug = model(data_transforms_boost(data))
          # pred_aug_score, pred_aug = output_aug.data.max(1, keepdim=False)
          # cls_aug = classes_to_names[int(pred_aug.item())].split(".")[1]
          # if cls_aug.lower() in f.lower():
          #   correct_aug+=1

          # output_attention = model(data_attention)
          output_attention = model(data_attention)
          pred_attention_score, pred_attention = output_attention.data.max(1, keepdim=False)
          cls_attention = classes_to_names[int(pred_attention.item())].split(".")[1]
          if cls_attention.lower() in f.lower():
            correct_attention+=1
          

          output_ensemble = torch.cat((pred, pred_crop, pred_attention) , dim=0)
          output_ensemble_score = torch.cat((pred_score, pred_crop_score, pred_attention_score) , dim=0)

          # output_ensemble = (pred +  pred_crop +  pred_attention)/3
          max_idx = output_ensemble_score.max(0)[1].item()
          # pred_ensemble = output_ensemble_score.max(0)[0].item()
          pred_ensemble = output_ensemble[max_idx]
          

          
          cls_ensemble = classes_to_names[int(pred_ensemble)].split(".")[1]
          if cls_ensemble.lower() in f.lower():
            correct_ensemble+=1

print("correct", correct, "correct_attention", correct_attention, "correct_crop", correct_crop, "correct_ensemble", correct_ensemble)
print("correct", correct/len_data ,"correct_attention", correct_attention/len_data, "correct_crop", correct_crop/len_data, "correct_ensemble", correct_ensemble/len_data)
  