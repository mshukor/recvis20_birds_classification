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

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

if use_cuda:
  state_dict = torch.load(args.model)
else:
  state_dict = torch.load(args.model, map_location=torch.device('cpu'))

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)


# model = inception_v3(pretrained=False)
# model.fc = nn.Linear(2048, 20)
if args.model2:
  model2 = torchvision.models.resnext50_32x4d(pretrained=True)
  model2.fc = nn.Linear(2048, 20)
  state_dict2 = torch.load(args.model2)
  model2.load_state_dict(state_dict2)
  model2.eval()
  model2.cuda()

model.load_state_dict(state_dict)
model.eval()

if use_cuda:
    print('Using GPU')
    model.cuda()
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


for d in tqdm(os.listdir(test_dir)):
    for f in tqdm(os.listdir(os.path.join(test_dir, d))):
      if 'jpg' in f:
          len_data+=1
          data = data_transforms(pil_loader(os.path.join(test_dir, d) + '/' + f))
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

          

          # output_mask = model(data_mask)
          # pred_mask_score, pred_mask = output_mask.data.max(1, keepdim=False)
          # cls_mask = classes_to_names[int(pred_mask.item())].split(".")[1]
          # if cls_mask.lower() in f.lower():
          #   correct_mask+=1
          if args.model2:
            output_mask = model2(data)
            pred_mask_score, pred_mask = output_mask.data.max(1, keepdim=False)
            cls_mask = classes_to_names[int(pred_mask.item())].split(".")[1]
            if cls_mask.lower() in f.lower():
              correct_mask+=1



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

          output_attention = model(data_attention)
          pred_attention_score, pred_attention = output_attention.data.max(1, keepdim=False)
          cls_attention = classes_to_names[int(pred_attention.item())].split(".")[1]
          if cls_attention.lower() in f.lower():
            correct_attention+=1
          

          output_ensemble = torch.cat((pred, pred_mask, pred_crop, pred_attention) , dim=0)
          output_ensemble_score = torch.cat((pred_score, pred_mask_score, pred_crop_score, pred_attention_score) , dim=0)

          max_idx = output_ensemble_score.max(0)[1].item()
          pred_ensemble = output_ensemble[max_idx]
          

          # output_ensemble = ( output_crop + output_mask + output) / 3
          # pred_ensemble = output_ensemble.data.max(1, keepdim=False)[1]
          
          cls_ensemble = classes_to_names[int(pred_ensemble.item())].split(".")[1]
          if cls_ensemble.lower() in f.lower():
            correct_ensemble+=1

print("correct", correct, "correct_attention", correct_attention, "correct_mask", correct_mask, "correct_crop", correct_crop, "correct_ensemble", correct_ensemble)
print("correct", correct/len_data ,"correct_attention", correct_attention/len_data, "correct_mask", correct_mask/len_data, "correct_crop", correct_crop/len_data, "correct_ensemble", correct_ensemble/len_data)
  