from efficientnet_pytorch import EfficientNet
import torch 
from data import data_transforms_train, data_transforms_val
from torchvision import datasets
import torch.nn as nn
import argparse
from data import ConcatDataset
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--model', type=str, default='inceptionv3_model_13.pth', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

args = parser.parse_args()


DATA = args.data
VALID_IMAGES = '/Inat_mini2' #Inat_mini2 val_images
TRAIN_IMAGES = '/train_images' # '/train_images' '/images

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

val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=2, shuffle=False, num_workers=1)

print(data_val.classes)
# model = inception_v3(pretrained=False)
SEMI = False
if not SEMI:
  model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
else:
  model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=256)

if use_cuda:
  checkpoint = torch.load(model_path)
else:
  checkpoint = torch.load(model_path, map_location=torch.device('cpu'))


# model.fc = nn.Linear(2048, 20)


model.load_state_dict(checkpoint) 

if SEMI:
  checkpoint1 = torch.load('experiment/eff6_ae_rotclass_model_11.pth')

  classifier =  nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
          )
  classifier.load_state_dict(checkpoint1) 



if use_cuda:
  model.cuda()
  if SEMI:
    classifier.cuda()
def validation():
    model.eval()
    if SEMI:
     classifier.eval()

    validation_loss = 0
    correct = 0
    for indata, target in tqdm(val_loader):
        if use_cuda:
            indata, target = indata.cuda(), target.cuda()
        output = model(indata)
        if SEMI:
          output = classifier(output)
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