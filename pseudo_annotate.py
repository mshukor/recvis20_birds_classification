from efficientnet_pytorch import EfficientNet
import torch 
from data import data_transforms_train, data_transforms_val
from torchvision import datasets
from inception import *
import torch.nn as nn
import argparse
from torchvision import datasets
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--model', type=str, default='inceptionv3_model_13.pth', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--model2', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

parser.add_argument('--dir', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--out', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--out-no', type=str, default='pseudo_no', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

parser.add_argument('--thresh', type=float, default=0.8, metavar='D',
                    help="threshold for pseudo labels")
parser.add_argument('--temp', type=float, default=1, metavar='D',
                    help="threshold for pseudo labels")
parser.add_argument('--test', type=bool, default=False, metavar='D',
                    help="threshold for pseudo labels")
args = parser.parse_args()


def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

DATA = args.dir
 
TEST_IMAGES = '/images' # '/test_images' 'train_images
model_path = args.model
model_path2 = args.model2
RESIZE = 500 #None

use_cuda = True
output_folder_pseudo = args.out
output_folder_pseudo_not = args.out_no

TRAIN_IMAGES = '/train_images' # '/train_images' '/images
data_orig = datasets.ImageFolder("bird_dataset" + TRAIN_IMAGES)
classes_to_names = {v: k for k, v in data_orig.class_to_idx.items()}

if RESIZE:
  data_transforms= transforms.Compose([
      transforms.Resize((RESIZE, RESIZE)),
      transforms.ToTensor(),
  ])
else:
   data_transforms= transforms.Compose([
      transforms.ToTensor(),
  ]) 


loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATA+ TEST_IMAGES, transform = data_transforms),
    batch_size=1, shuffle=False, num_workers=1)

# model = inception_v3(pretrained=False)
model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
if use_cuda:
  checkpoint = torch.load(model_path)
else:
  checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint) 
if model_path2:
  model2 = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
  if use_cuda:
    checkpoint2 = torch.load(model_path2)
  else:
    checkpoint2 = torch.load(model_path2, map_location=torch.device('cpu'))
  model2.load_state_dict(checkpoint2) 

# model.fc = nn.Linear(2048, 20)



softmax = torch.nn.Softmax(dim=1)

T = args.temp

if use_cuda:
  model.cuda()
  if args.model2:
    model2.cuda()
def pseudo_annotate():
    model.eval()
    if args.model2:
      model2.eval()
    count = 0
    count0 = 0
    count2 = 0
    count_diff = 0
    for data, target in tqdm(loader):
        if args.model2:
          print("count0", count0, "count2", count2, "count_diff", count_diff)
        if use_cuda:
            data, target = data.cuda(), target.cuda()       
        if args.model2:
          output2 = model2(data)
          pred2 = output2.data
          confidence2 = softmax(pred2/T)*100.
          confidence2, cls2 = confidence2.max(1, keepdim=True)
        output = model(data)
        pred = output.data
        confidence0 = softmax(pred/T)*100.
        confidence0, cls0 = confidence0.max(1, keepdim=True)
        if args.model2:
          if cls0[0].item() != cls2[0].item():
            count_diff+=1
          if confidence0[0] > confidence2[0]:
            print(cls0[0].item(), confidence0[0].item(), ">", confidence2[0].item(), cls2[0].item())
            confidence = confidence0
            best_class = cls0[0].item()
            count0+=1
          else:
            print(cls0[0].item(), confidence0[0].item(), "<", confidence2[0].item(), cls2[0].item())
            confidence = confidence2
            best_class = cls2[0].item()
            count2+=1
        else:
          confidence = confidence0
          best_class = cls0[0].item()

        folder_name = classes_to_names[best_class]

        os.makedirs(output_folder_pseudo, exist_ok = True)
        os.makedirs(output_folder_pseudo+'/'+ 'train_images'+'/'+ folder_name, exist_ok = True)

        os.makedirs(output_folder_pseudo_not, exist_ok = True)
        os.makedirs(output_folder_pseudo_not+'/'+ 'train_images'+'/'+ folder_name, exist_ok = True)
        if use_cuda:
          image = np.rollaxis(data[0].cpu().numpy(), 0, 3)
        else:
          image = np.rollaxis(data[0].numpy(), 0, 3)
        # image = Image.fromarray(normalize8(image)).convert('RGB')
        # image = cv2.cvtColor(normalize8(image), cv2.COLOR_BGR2RGB)
        image = normalize8(image)
        print(confidence[0].item(), count)
        if confidence[0] > args.thresh*100:
          count +=1
          print(count)
          if not args.test:
            plt.imsave(output_folder_pseudo+'/'+'train_images'+'/'+folder_name+'/'+folder_name.split(".")[1] + "_" + str(count) + ".jpg", image, dpi=1000)
        else:
          if not args.test:
            plt.imsave(output_folder_pseudo_not+'/'+'train_images'+'/'+folder_name+'/'+folder_name.split(".")[1] + "_" + str(count) + ".jpg", image, dpi=1000)


    print('\n Number of annotated images: {}, percentage_annotated = {}/{}'.format(
        count, 100. * count / len(loader.dataset)))
    
pseudo_annotate()