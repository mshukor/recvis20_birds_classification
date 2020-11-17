import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from inception import *

from model import Net
import torch.nn as nn

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--data-crop', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--data-attention', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")                    
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

CHANNELS = "TRIPLE"


if use_cuda:
  state_dict = torch.load(args.model)
else:
  state_dict = torch.load(args.model, map_location=torch.device('cpu'))

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)

if CHANNELS == "DOUBLE":
  model._conv_stem = Conv2dStaticSamePadding(in_channels=3*2, out_channels=56, kernel_size=(3, 3), stride=2, image_size=(456, 456))
if CHANNELS == "TRIPLE":
  model._conv_stem = Conv2dStaticSamePadding(in_channels=3*3, out_channels=56, kernel_size=(3, 3), stride=2, image_size=(456, 456))



# model = inception_v3(pretrained=False)
# model.fc = nn.Linear(2048, 20)

model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

from data import data_transforms_val

test_dir = args.data + '/test_images/mistery_category'
test_dir_crop = args.data + '/test_images/mistery_category'
test_dir_attention = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

data_transforms = data_transforms_val
output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))

        data_crop = data_transforms(pil_loader(test_dir_crop + '/' + f))
        data_crop = data_crop.view(1, data.size(0), data.size(1), data.size(2))

        data_attention = data_transforms(pil_loader(test_dir_attention + '/' + f))
        data_attention = data_attention.view(1, data.size(0), data.size(1), data.size(2))

        sample = torch.cat((data, data_crop, data_attention), dim=0)

        if use_cuda:
            sample = sample.cuda()
        output = model(sample)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        


