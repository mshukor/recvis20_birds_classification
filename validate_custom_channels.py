from efficientnet_pytorch import EfficientNet
import torch 
from data import data_transforms_train, data_transforms_val
from torchvision import datasets
from inception import *
import torch.nn as nn
import argparse
from data import ConcatDataset
from torchvision import datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from data import DoubleChannels, TripleChannels
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--model', type=str, default='inceptionv3_model_13.pth', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-crop', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-attention', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

args = parser.parse_args()


DATA = args.data
VALID_IMAGES = '/val_images' #
TRAIN_IMAGES = '/train_images' # '/train_images' '/images
CHANNELS = "TRIPLE"
model_path = args.model
use_cuda = True


torch.manual_seed(0)

NEW_EVAL = False

def train_val_dataset(dataset, val_split=0.055):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, val_idx)
    return dataset_train, dataset_val

if NEW_EVAL:
  data_old_train = datasets.ImageFolder(DATA + TRAIN_IMAGES, transform=data_transforms_train)
  data_old_val = datasets.ImageFolder(DATA + VALID_IMAGES, transform=data_transforms_train)
  dataset = ConcatDataset(data_old_train, data_old_val)
  print(len(dataset))
  _, data_val = train_val_dataset(dataset)

else:
  data_val = datasets.ImageFolder(DATA+ VALID_IMAGES,
                        transform=data_transforms_val)

data_combined_val = TripleChannels(args.data + VALID_IMAGES, args.data_crop + VALID_IMAGES, 
args.data_attention + VALID_IMAGES, transform = data_transforms_val, same = True)

val_loader = torch.utils.data.DataLoader(
      data_combined_val,
      batch_size=2, shuffle=False, num_workers=1)


# val_loader = torch.utils.data.DataLoader(
#     data_val,
#     batch_size=2, shuffle=False, num_workers=1)


# model = inception_v3(pretrained=False)
model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
if CHANNELS == "DOUBLE":
  model._conv_stem = Conv2dStaticSamePadding(in_channels=3*2, out_channels=56, kernel_size=(3, 3), stride=2, image_size=(456, 456))
if CHANNELS == "TRIPLE":
  model._conv_stem = Conv2dStaticSamePadding(in_channels=3*3, out_channels=56, kernel_size=(3, 3), stride=2, image_size=(456, 456))


if use_cuda:
  checkpoint = torch.load(model_path)
else:
  checkpoint = torch.load(model_path, map_location=torch.device('cpu'))


# model.fc = nn.Linear(2048, 20)

model.load_state_dict(checkpoint) 

if use_cuda:
  model.cuda()
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for indata, target in val_loader:
        if use_cuda:
            indata, target = indata.cuda(), target.cuda()
        output = model(indata)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
validation()