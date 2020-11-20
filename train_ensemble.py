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
from data import DoubleChannels, TripleChannels
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

# class EFFICIENT(nn.Module):
#     def __init__(self, num_classes):
#         super(EFFICIENT, self).__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=512)
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.classifier = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#     def forward(self, x):
#         out = self.model(x)
#         # out = out.view(x.size(0), -1)
#         out = self.relu (self.fc1(out))
#         out = self.dropout(out)
#         out = self.relu (self.fc2(out))
#         out = self.dropout(out)
#         out = self.classifier(out)
#         return out

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
parser.add_argument('--data-attention', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-pseudo-2', type=str, default=None, metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--data-nabirds', type=str, default=None, metavar='D',
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
parser.add_argument('--weight-decay', default=2e-4, type=float, help='weight decay')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

torch.manual_seed(0)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms_train, data_transforms_val
MODEL = "EFFICIENT" # EFFICIENT INCEPTION INCEPTIONRESNETV2 VIT BIT RESNEXT
FREEZE = True
TRAIN_IMAGES = '/train_images' # '/train_images' '/images
VALID_IMAGES = '/val_images' #
VALID = True
PRETRAIN = False
CHANNELS = "SINGLE" # "TRIPLE"
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

if MODEL == "VIT":
  data_transforms_train = transforms.Compose([
      transforms.Resize((384, 384)),
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(30),
      transforms.RandomResizedCrop((384, 384)),

      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
  ])

  data_transforms_val = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
  ])


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
if CHANNELS == "DOUBLE":
  data_combined = DoubleChannels(args.data + TRAIN_IMAGES, args.data_crop + TRAIN_IMAGES, transform = data_transforms_val)
  data_combined_val = DoubleChannels(args.data + VALID_IMAGES, args.data_crop + VALID_IMAGES, transform = data_transforms_val)
elif CHANNELS == "TRIPLE":
  data_combined = TripleChannels(args.data + TRAIN_IMAGES, args.data_crop + TRAIN_IMAGES, args.data_attention + TRAIN_IMAGES, transform = data_transforms_val)
  data_combined_val = TripleChannels(args.data + VALID_IMAGES, args.data_crop + VALID_IMAGES, args.data_attention + VALID_IMAGES, transform = data_transforms_val)

else:
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

  if args.data_attention:
    data_attention = datasets.ImageFolder(args.data_attention + TRAIN_IMAGES,
                          transform=data_transforms_val)
  else:
    data_attention = None
  if args.data_pseudo_2:
    data_pseudo_2 = datasets.ImageFolder(args.data_pseudo_2 + TRAIN_IMAGES,
                          transform=data_transforms_val)
  else:
    data_pseudo_2 = None

  if args.data_nabirds:
    data_nabirds = datasets.ImageFolder(args.data_nabirds + '/train_images',
                          transform=data_transforms_val)
  else:
    data_nabirds = None



if CHANNELS != "DOUBLE" and CHANNELS != "TRIPLE":



  if args.data_nabirds and args.data_crop and args.data_attention and args.data_pseudo and args.data_pseudo_2:
    train_data = ConcatDataset(data_orig, data_crop, data_attention, data_pseudo, data_pseudo_2, data_nabirds)
   
    targets += data_crop.targets
    targets +=  data_attention.targets
    targets +=  data_pseudo.targets
    targets +=  data_nabirds.targets

  elif args.data_crop and args.data_attention and args.data_pseudo :
    train_data = ConcatDataset(data_orig, data_crop, data_attention, data_pseudo)
   
    targets += data_crop.targets
    targets +=  data_attention.targets
    targets +=  data_pseudo.targets

  elif args.data_crop and args.data_mask:
    train_data = ConcatDataset(data_orig, data_crop, data_mask)

  elif args.data_crop and args.data_pseudo:
    train_data = ConcatDataset(data_orig, data_crop, data_pseudo)
    targets += data_crop.targets
    targets +=  data_pseudo.targets

  elif args.data_crop and args.data_attention:
    train_data = ConcatDataset(data_orig, data_crop, data_attention)
    targets += data_crop.targets
    targets +=  data_attention.targets

  elif args.data_crop:
    train_data = ConcatDataset(data_orig, data_crop)
    targets += data_crop.targets
  else:
    train_data = data_orig


if BALANCE_CLASSES:
  class_sample_count = torch.unique(torch.from_numpy(np.array(targets)), return_counts=True)[1]
  print(class_sample_count)
  weight = 1. / np.array(class_sample_count)
  samples_weight = np.array([weight[t] for t in targets])
  samples_weight = torch.from_numpy(samples_weight)
  samples_weigth = samples_weight.double()
  sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

  print(weight)
  shuffle = False
else:
  sampler = None
  shuffle = True
# classes_to_names = {v: k for k, v in data_old_train.class_to_idx.items()}
# print(classes_to_names)
if CHANNELS == "DOUBLE" or CHANNELS == "TRIPLE":
  train_loader = torch.utils.data.DataLoader(
      data_combined,
      batch_size=args.batch_size, num_workers=1, sampler=sampler, shuffle=shuffle) #sampler=sampler 

  val_loader = torch.utils.data.DataLoader(
        data_combined_val,
        batch_size=2, shuffle=False, num_workers=1)
else:
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size, num_workers=1, sampler=sampler, shuffle=shuffle) #sampler=sampler
  if VALID:
    val_loader = torch.utils.data.DataLoader(
        data_orig_val,
        batch_size=1, shuffle=False, num_workers=1)
    if args.data_crop:
      val_loader_crop = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data_crop + VALID_IMAGES,
                            transform=data_transforms_val),
        batch_size=2, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script


if MODEL == "MIX":
    backbone_generalized = EfficientNet.from_pretrained('efficientnet-b6', num_classes=555)
    backbone_specialized = EfficientNet.from_pretrained('efficientnet-b6', num_classes=555)
    model_head = nn.Sequential(
          nn.Linear(555*2, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, args.num_classes),
        )
    for name, param in backbone_generalized.named_parameters():
      if param.requires_grad:
          param.requires_grad = False

elif MODEL == "EFFICIENT":
  if PRETRAIN:
    model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=555)
    checkpoint = torch.load("experiment/efficient_pretrain_nabirds_model_1.pth")
    model.load_state_dict(checkpoint) 
    model._fc = nn.Linear(2048, args.num_classes)
  else:
    # model = EFFICIENT(args.num_classes)
    model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=8)
    model2 = EfficientNet.from_pretrained('efficientnet-b6', num_classes=7)
    model3 = EfficientNet.from_pretrained('efficientnet-b6', num_classes=8)
    # if args.model:
    #     print("loading pretrained model")
    #     checkpoint = torch.load(args.model)
    #     model.load_state_dict(checkpoint) 

    if CHANNELS == "DOUBLE":
      model._conv_stem = Conv2dStaticSamePadding(in_channels=3*2, out_channels=56, kernel_size=(3, 3), stride=2, image_size=(456, 456))
    if CHANNELS == "TRIPLE":
      model._conv_stem = Conv2dStaticSamePadding(in_channels=3*3, out_channels=56, kernel_size=(3, 3), stride=2, image_size=(456, 456))


  #   model._fc = nn.Sequential(
  #         nn.Linear(2304, 256),
  #         nn.ReLU(),
  #         nn.Linear(256, 128),
  #         nn.ReLU(),
  #         nn.Linear(128, args.num_classes),
  #       )

  if FREEZE:
    for name, param in model.named_parameters():
      if name == '_blocks.43._bn2.bias':
        break
      if param.requires_grad:
          param.requires_grad = False
    for name, param in model2.named_parameters():
      if name == '_blocks.43._bn2.bias':
        break
      if param.requires_grad:
          param.requires_grad = False
    for name, param in model3.named_parameters():
      if name == '_blocks.43._bn2.bias':
        break
      if param.requires_grad:
          param.requires_grad = False

elif MODEL == "RESNEXT":
  model = torchvision.models.resnext50_32x4d(pretrained=True)
  model.fc = nn.Linear(2048, args.num_classes)
  if FREEZE:
    for name, param in model.named_parameters():
      if 'fc' in name:
        break
      if param.requires_grad:
          param.requires_grad = False

elif MODEL == "INCEPTION":
  model = inception_v3(pretrained=False)
  model.fc = nn.Linear(2048, 8142)
  print("loading pretrained model")
  checkpoint = torch.load("iNat_2018_InceptionV3.pth.tar")
  model.load_state_dict(checkpoint['state_dict']) 
  model.fc = nn.Linear(2048, args.num_classes)
  model.aux_logits = False
  if FREEZE:
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.split(".")[0] not in ["Mixed_6e", "AuxLogits", "Mixed_7a", "Mixed_7b", "Mixed_7c", "fc"]:
              param.requires_grad = False
elif MODEL == "INCEPTIONRESNETV2":
  model_name = 'inceptionresnetv2' # could be fbresnet152 or inceptionresnetv2
  model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
  model.last_linear = nn.Linear(1536, args.num_classes, bias = True)
elif MODEL=="VIT":
  model = timm.create_model('vit_large_patch16_384', pretrained=True)
  model.head = nn.Linear(1024, args.num_classes, bias = True)
elif MODEL == "BIT":
  model = models.KNOWN_MODELS['BiT-M-R101x1'](head_size= args.num_classes, zero_head=True)
  model.load_from(np.load('BiT-M-R101x1.npz'))
  

print("USING : ", CHANNELS)

if args.model and CHANNELS != 'TRIPLE' and CHANNELS != 'DOUBLE':
    print("loading pretrained model")
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint) 
 
from model import Net
# model = Net()
if use_cuda:
    print('Using GPU')
    if MODEL == "MIX":
      model_head.cuda()
      backbone_generalized.cuda()
      backbone_specialized.cuda()
    else:
      model.cuda()
else:
    print('Using CPU')

if MODEL == "MIX":
  optimizer = RangerAdaBelief(list(backbone_specialized.parameters()) + list(model_head.parameters()), lr=args.lr, eps=1e-12, betas=(0.9,0.999))
else:
  # optimizer = optim.Adam(model.parameters(), lr=args.lr) #momentum=args.momentum

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay) #momentum=args.momentum
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
  optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay) #momentum=args.momentum
  lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.epochs)
  optimizer3 = optim.SGD(model3.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay) #momentum=args.momentum
  lr_scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, args.epochs)

  # optimizer = AdaBelief(model.parameters(), lr=args.lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
  # optimizer = RangerAdaBelief(model.parameters(), lr=args.lr, eps=1e-12, betas=(0.9,0.999))
  # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


def train(epoch):
    lr_scheduler.step()
    lr_scheduler2.step()
    lr_scheduler3.step()
    model.train()
    model2.train()
    model3.train()
    correct = 0
    i = 0 
    for batch_idx, (data, target) in enumerate(train_loader):
        i+=1
        if target.numpy().any() >= 20 and target.numpy().any() < 0:
            print(target.numpy())
            continue
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        
        
        output = model(data)
        output2 = model2(data)
        output3 = model3(data)
      
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        criterion2 = torch.nn.CrossEntropyLoss(reduction='mean')
        criterion3 = torch.nn.CrossEntropyLoss(reduction='mean')

        target = target
        mask = target.le(6).long() # * target.ge(1).long()
        mask2 = target.le(12).long() * target.ge(7).long()
        mask3 = target.ge(13).long()

     
        loss = criterion(output, target * mask)
        loss2 = criterion2(output2, (target -6) * mask2)
        loss3 = criterion3(output3, (target -12) * mask3)

        # loss.requres_grad = True

        loss.backward()
        optimizer.step()

        loss2.backward()
        optimizer2.step()

        loss3.backward()
        optimizer3.step()

        # pred = output.data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # if i > 10:
        #   break
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Loss2: {:.6f} Loss3: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), loss2.data.item(), loss3.data.item()))

def validation(val_loader):
    model.eval()
    model2.eval()
    model3.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        output2 = model2(data)
        output3 = model3(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        score, pred = output.data.max(1, keepdim=True)
        score2, pred2 = output2.data.max(1, keepdim=True)
        score3, pred3 = output3.data.max(1, keepdim=True)

        
        output_ensemble = torch.cat((pred , pred2, pred3 ) , dim=0)
        mask = output_ensemble.gt(0)
     
        output_ensemble_score = torch.cat((score, score2, score3) , dim=0)
        output_ensemble_score = output_ensemble_score * mask

        max_idx = output_ensemble_score.max(0)[1].item()
    
        max_add = [-1, 8 -1, 13 -1]
      
    
        pred_ensemble = output_ensemble[max_idx] + max_add[max_idx]

        correct += pred.eq(target.data.view_as(pred_ensemble)).cpu().sum()

    # validation_loss /= len(val_loader.dataset)
    print('\n Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))



model_name = "/" + args.name
model.cuda()
model2.cuda()
model3.cuda()

for epoch in range(1, args.epochs + 1):
    if MODEL == "MIX":
      train_mix(epoch)
    else:
      train(epoch)
    if VALID:
      if MODEL == "MIX":
        validation_mix(val_loader)
      else:
        validation(val_loader)
      if args.data_crop and CHANNELS != "DOUBLE" and CHANNELS != "TRIPLE":
        print("validation on cropped")
        if MODEL == "MIX":
          validation_mix(val_loader_crop)
        else:
          validation(val_loader_crop)
    if MODEL == "MIX":
      model_file_back = args.experiment + model_name +'_back_' + '_model_' + str(epoch) + '.pth'
      torch.save(backbone_specialized.state_dict(), model_file_back)

      model_file_head = args.experiment + model_name +'_head_' + '_model_' + str(epoch) + '.pth'
      torch.save(model_head.state_dict(), model_file_head)
      print('Saved model to ' + model_file_back + '. You can run `python evaluate.py --model ' + model_file_back + '` to generate the Kaggle formatted csv file\n')
      print('Saved model to ' + model_file_head + '. You can run `python evaluate.py --model ' + model_file_head + '` to generate the Kaggle formatted csv file\n')
    else:
      
      model_file = args.experiment + model_name +  '_model_1_' + str(epoch) + '.pth'
      torch.save(model.state_dict(), model_file)
      print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')

      model_file2 = args.experiment + model_name +  '_model_2_' + str(epoch) + '.pth'
      torch.save(model.state_dict(), model_file2)
      print('Saved model to ' + model_file2 + '. You can run `python evaluate.py --model ' + model_file2 + '` to generate the Kaggle formatted csv file\n')

      model_file3 = args.experiment + model_name +  '_model_3_' + str(epoch) + '.pth'
      torch.save(model.state_dict(), model_file3)
      print('Saved model to ' + model_file3 + '. You can run `python evaluate.py --model ' + model_file3 + '` to generate the Kaggle formatted csv file\n')
