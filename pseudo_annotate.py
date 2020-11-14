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
parser.add_argument('--dir', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--out', type=str, default='bird_dataset', metavar='D',
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

TEST_IMAGES = '/test_images' # '/test_images'
model_path = args.model
use_cuda = False
output_folder_pseudo = args.out


TRAIN_IMAGES = '/train_images' # '/train_images' '/images
data_orig = datasets.ImageFolder("bird_dataset" + TRAIN_IMAGES)
classes_to_names = {v: k for k, v in data_orig.class_to_idx.items()}

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


# model.fc = nn.Linear(2048, 20)

model.load_state_dict(checkpoint) 

softmax = torch.nn.Softmax(dim=1)

T = args.temp

if use_cuda:
  model.cuda()
def pseudo_annotate():
    model.eval()
    count = 0
    for data, target in tqdm(loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # get the index of the max log-probability
        pred = output.data
        confidence = softmax(pred/T)*100.
        confidence, cls = confidence.max(1, keepdim=True)
        folder_name = classes_to_names[cls[0].item()]

        os.makedirs(output_folder_pseudo, exist_ok = True)
        os.makedirs(output_folder_pseudo+'/'+ 'train_images'+'/'+ folder_name, exist_ok = True)

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


    print('\n Number of annotated images: {}, percentage_annotated = {}/{}'.format(
        count, 100. * count / len(loader.dataset)))
    
pseudo_annotate()