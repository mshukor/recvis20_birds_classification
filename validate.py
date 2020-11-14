from efficientnet_pytorch import EfficientNet
import torch 
from data import data_transforms_train, data_transforms_val
from torchvision import datasets
from inception import *
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--model', type=str, default='inceptionv3_model_13.pth', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

args = parser.parse_args()


DATA = 'bird_dataset'
VALID_IMAGES = '/val_images' #
model_path = args.model
use_cuda = False

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATA+ VALID_IMAGES,
                        transform=data_transforms_val),
    batch_size=4, shuffle=False, num_workers=1)

# model = inception_v3(pretrained=False)
model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=20)
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
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
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