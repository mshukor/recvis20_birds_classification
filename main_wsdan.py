"""TRAINING
Inspired from : 
Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
https://github.com/GuYuc/WS-DAN.PyTorch/blob/master/train.py
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief
from inception import *
from data import ConcatDataset
import pretrainedmodels
from sklearn.model_selection import train_test_split
import numpy as np
import timm
import torchvision.transforms as transforms
import bit_torch_models as models
# from models import EFFICIENT
import torch.nn.functional as F
import torchvision
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment

from models import WSDAN

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-crop', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-mask', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-pseudo', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

parser.add_argument('--model', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--name', type=str, default='efficient', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--num_classes', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--online_da', type=bool, default=True, metavar='N',
                    help='online data augmentaiion')
# parser.add_argument('--merged', type=bool, default=False, metavar='N',
#                     help='use several datasets')


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

torch.manual_seed(0)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms_train, data_transforms_val

TRAIN_IMAGES = '/train_images' # '/train_images' '/images
VALID_IMAGES = '/val_images' #
VALID = True

num_attentions = 32   
NET = 'resnet34' #'"inception_mixed_6e" 
BETA = 5e-2  
NEW_EVAL = False
BALANCE_CLASSES = False

if args.online_da:
  train_transform = data_transforms_val
else:
  train_transform = data_transforms_train
  
def train_val_dataset(dataset, val_split=0.055):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    dataset_train = torch.utils.data.Subset(dataset, train_idx)
    dataset_val = torch.utils.data.Subset(dataset, val_idx)
    return dataset_train, dataset_val



if NEW_EVAL:
  data_old_train = datasets.ImageFolder(args.data + TRAIN_IMAGES, transform=data_transforms_train)
  data_old_val = datasets.ImageFolder(args.data + VALID_IMAGES, transform=data_transforms_train)
  dataset = ConcatDataset(data_old_train, data_old_val)
  print(len(dataset))
  data_orig, data_orig_val = train_val_dataset(dataset)
  targets = [t[1] for t in tqdm(data_orig)]
else:
  data_orig = datasets.ImageFolder(args.data + TRAIN_IMAGES,
                            transform=data_transforms_train)
  data_orig_val = datasets.ImageFolder(args.data + VALID_IMAGES,
                            transform=data_transforms_val)
  targets = data_orig.targets
if args.data_crop:
  data_crop = datasets.ImageFolder(args.data_crop + TRAIN_IMAGES,
                          transform=data_transforms_val)
else:
  data_crop = None

if args.data_mask:
  data_mask = datasets.ImageFolder(args.data_mask + TRAIN_IMAGES,
                        transform=data_transforms_val)
else:
  data_mask = None

if args.data_pseudo:
  data_pseudo = datasets.ImageFolder(args.data_pseudo + TRAIN_IMAGES,
                        transform=data_transforms_val)
else:
  data_pseudo = None





if args.data_crop and args.data_mask:
  train_data = ConcatDataset(data_orig, data_crop, data_mask)

elif args.data_crop and args.data_pseudo:
  train_data = ConcatDataset(data_orig, data_crop, data_pseudo)
  targets += data_crop.targets
  targets +=  data_pseudo.targets

elif args.data_crop:
  train_data = ConcatDataset(data_orig, data_crop)
  targets += data_crop.targets
else:
  train_data = data_orig


train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, num_workers=1, shuffle = True) #sampler=sampler


if VALID:
  val_loader = torch.utils.data.DataLoader(
      data_orig_val,
      batch_size=2, shuffle=False, num_workers=1)
  if args.data_crop:
    val_loader_crop = torch.utils.data.DataLoader(
      datasets.ImageFolder(args.data_crop + VALID_IMAGES,
                          transform=data_transforms_val),
      batch_size=2, shuffle=False, num_workers=1)

device = torch.device("cuda")
print("define wsdan")
model = WSDAN(num_classes=args.num_classes, M=num_attentions, net=NET, pretrained=True)
feature_center = torch.zeros(args.num_classes, num_attentions * model.num_features).to(device)
center_loss = CenterLoss()
cross_entropy_loss = nn.CrossEntropyLoss()

if args.model:
    print("loading pretrained model")
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint) 
 
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')


# optimizer = optim.Adam(model.parameters(), lr=args.lr) #momentum=args.momentum

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #momentum=args.momentum
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# optimizer = AdaBelief(model.parameters(), lr=args.lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
optimizer = RangerAdaBelief(model.parameters(), lr=args.lr, eps=1e-12, betas=(0.9,0.999))
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


def train(epoch):
    # lr_scheduler.step()
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if target.numpy().any() >= 20 and target.numpy().any() < 0:
            print(target.numpy())
            continue
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        y_pred_raw, feature_matrix, attention_map = model(data)
        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[target], dim=-1)
        feature_center[target] += BETA * (feature_matrix.detach() - feature_center_batch)
      
        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(data, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

        # crop images forward
        y_pred_crop, _, _ = model(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(data, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

        # drop images forward
        y_pred_drop, _, _ = model(drop_images)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, target) / 3. + \
                     cross_entropy_loss(y_pred_crop, target) / 3. + \
                     cross_entropy_loss(y_pred_drop, target) / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()
      
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss.data.item()))

def validation(val_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        ##################################
        # Raw Image
        ##################################
        y_pred_raw, _, attention_map = model(data)

        ##################################
        # Object Localization and Refinement
        ##################################
        crop_images = batch_augment(data, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _ = model(crop_images)

        ##################################
        # Final prediction
        ##################################
        y_pred = (y_pred_raw + y_pred_crop) / 2.

        # loss
        batch_loss = cross_entropy_loss(y_pred, target)

        # metrics: top-1,5 error
        # epoch_acc = raw_metric(y_pred, y)

        # get the index of the max log-probability
        pred = y_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    batch_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        batch_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))



model_name = "/" + args.name
for epoch in range(1, args.epochs + 1):
    train(epoch)
    if VALID:
      validation(val_loader)
      if args.data_crop:
        print("validation on cropped")
        validation(val_loader_crop)
    
    model_file = args.experiment + model_name +  '_model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
