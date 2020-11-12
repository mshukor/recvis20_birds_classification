import random
import os
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from tqdm import tqdm

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

data_transforms_train = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop((456, 456), scale=(0.09, 1.0), ratio=(0.9, 1.1)),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

dirName = 'bird_dataset/train_images'

listOfFile = os.listdir(dirName)
completeFileList = list()
for file in tqdm(listOfFile):
    completePath = os.path.join(dirName, file)
    image_paths = os.listdir(completePath)
    for image in image_paths:
      try:
        img_path = os.path.join(completePath, image)
        img= Image.open(img_path)
        img_aug= data_transforms_train(img)
        im= np.rollaxis(img_aug.numpy(), 0, 3)
        im_pil = Image.fromarray(normalize8(im)).convert('RGB')

        prefix = ('/').join(img_path.split("/")[:-1])
        name = img_path.split("/")[-1].split(".")[0]

        im_pil.save(prefix +'/' + name + "_aug.jpg")
      except TypeError or ValueError:
        continue



