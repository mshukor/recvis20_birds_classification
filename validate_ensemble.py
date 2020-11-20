from efficientnet_pytorch import EfficientNet
import torch 
from data import data_transforms_train, data_transforms_val
from torchvision import datasets
from inception import *
import torch.nn as nn
import argparse
from data import ConcatDataset

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--model', type=str, default='inceptionv3_model_13.pth', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

args = parser.parse_args()


DATA = args.data
VALID_IMAGES = '/val_images' #
TRAIN_IMAGES = '/train_images' # '/train_images' '/images

model_path = args.model
model_path0 = 'experiment/resnet_0_model_15.pth'
model_path1 = 'experiment/resnet_1_model_8.pth'
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

val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=1, shuffle=False, num_workers=1)


# model = inception_v3(pretrained=False)
# model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)  
model.fc = nn.Linear(512, 20, bias = True)

model0 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)  
model0.fc = nn.Linear(512, 10, bias = True)

model1 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)  
model1.fc = nn.Linear(512, 10, bias = True)


if use_cuda:
  checkpoint = torch.load(model_path)
  checkpoint0 = torch.load(model_path0)
  checkpoint1 = torch.load(model_path1)
else:
  checkpoint = torch.load(model_path, map_location=torch.device('cpu'))


# model.fc = nn.Linear(2048, 20)

model.load_state_dict(checkpoint) 
model0.load_state_dict(checkpoint0) 
model1.load_state_dict(checkpoint1) 

if use_cuda:
  model.cuda()
  model0.cuda()
  model1.cuda()
def validation():
    model.eval()
    model0.eval()
    model1.eval()

    validation_loss = 0
    correct = 0
    correct_ensemble = 0
    for indata, target in val_loader:
        if use_cuda:
            indata, target = indata.cuda(), target.cuda()
        output = model(indata)
        output0 = model0(indata)
        output1 = model1(indata)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()


        score, pred = output.data.max(1, keepdim=True)
        score0, pred0 = output0.data.max(1, keepdim=True)
        score1, pred1 = output1.data.max(1, keepdim=True)

        
        output_ensemble = torch.cat((pred0 , pred1) , dim=0)
        # mask = output_ensemble.gt(0)
     
        output_ensemble_score = torch.cat((score0, score1) , dim=0)
        # output_ensemble_score = output_ensemble_score * mask

        max_idx = output_ensemble_score.max(0)[1].item()
    
        max_add = [0, 10]
      
        pred_ensemble = output_ensemble[max_idx] + max_add[max_idx]
        print(max_idx)
        print(pred_ensemble)
        print(target)
        correct_ensemble += pred_ensemble.eq(target.data.view_as(pred_ensemble)).cpu().sum()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    print('\n ENSEMBLE : Accuracy: {}/{} ({:.0f}%)'.format(
        correct_ensemble, len(val_loader.dataset),
        100. * correct_ensemble / len(val_loader.dataset)))
    
validation()