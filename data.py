import zipfile
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets
from randaugment import RandAugmentMC

data_transforms_train = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop((456, 456)),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_transforms_val = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)
    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
class DoubleChannels(torch.utils.data.Dataset):
  def __init__(self, dataset_path, concat_dataset_path, transform=None):

      self.dataset_path = dataset_path
      self.concat_dataset_path = concat_dataset_path

      data_orig = datasets.ImageFolder(dataset_path, transform=transform)
      data_concat = datasets.ImageFolder(concat_dataset_path, transform=transform)


      self.classes = data_orig.classes
      self.class_to_idx = data_concat.class_to_idx

      self.samples_images = data_orig.samples
      self.samples_image_concat = data_concat.samples
      self.transform = transform
      self.targets_images = data_orig.targets
      self.targets_image_concat = data_concat.targets

      concat_targets = np.array(self.targets_image_concat)

      self.orig_to_concat = {key: np.where(concat_targets == key) for key in range(len(self.classes))}
  def __len__(self) -> int:
          return len(self.samples_images)

  def __getitem__(self, index):

        path_image, target_image = self.samples_images[index]
        concat_index = np.random.choice(self.orig_to_concat[target_image][0], 1)
        # path_image_concat = path_image.replace(self.dataset_path, self.concat_dataset_path)
        path_image_concat, target_image_concat = self.samples_image_concat[concat_index[0]]
        sample_image =Image.open(path_image)
        sample_image_concat = Image.open(path_image_concat)

        if self.transform is not None:
            sample_image = self.transform(sample_image)
            sample_image_concat = self.transform(sample_image_concat)

        sample = torch.cat((sample_image, sample_image_concat), dim=0)

        return sample, target_image



class TripleChannels(torch.utils.data.Dataset):
  def __init__(self, dataset_path, concat_dataset_path, concat_dataset_path_2, transform=None, same=False):
      self.same = same
      self.dataset_path = dataset_path
      self.concat_dataset_path = concat_dataset_path
      self.concat_dataset_path_2 = concat_dataset_path_2

      data_orig = datasets.ImageFolder(dataset_path, transform=transform)
      data_concat = datasets.ImageFolder(concat_dataset_path, transform=transform)
      data_concat_2 = datasets.ImageFolder(concat_dataset_path_2, transform=transform)


      self.classes = data_orig.classes
      self.class_to_idx = data_concat.class_to_idx

      self.samples_images = data_orig.samples
      self.samples_image_concat = data_concat.samples
      self.samples_image_concat_2 = data_concat_2.samples
      self.transform = transform
      self.targets_images = data_orig.targets
      self.targets_image_concat = data_concat.targets
      self.targets_image_concat_2 = data_concat_2.targets

      concat_targets = np.array(self.targets_image_concat)
      concat_targets_2 = np.array(self.targets_image_concat_2)

      self.orig_to_concat = {key: np.where(concat_targets == key) for key in range(len(self.classes))}
      self.orig_to_concat_2 = {key: np.where(concat_targets_2 == key) for key in range(len(self.classes))}

  def __len__(self) -> int:
          return len(self.samples_images)

  def __getitem__(self, index):

        path_image, target_image = self.samples_images[index]
        concat_index = np.random.choice(self.orig_to_concat[target_image][0], 1)
        concat_index_2 = np.random.choice(self.orig_to_concat_2[target_image][0], 1)
        if self.same:
          path_image_concat = path_image.replace(self.dataset_path, self.concat_dataset_path)
          path_image_concat_2 = path_image.replace(self.dataset_path, self.concat_dataset_path_2)
        else:
          path_image_concat, target_image_concat = self.samples_image_concat[concat_index[0]]
          path_image_concat_2, target_image_concat_2 = self.samples_image_concat_2[concat_index_2[0]]

        sample_image =Image.open(path_image)
        sample_image_concat = Image.open(path_image_concat)
        sample_image_concat_2 = Image.open(path_image_concat_2)

        if self.transform is not None:
            sample_image = self.transform(sample_image)
            sample_image_concat = self.transform(sample_image_concat)
            sample_image_concat_2 = self.transform(sample_image_concat_2)

        sample = torch.cat((sample_image, sample_image_concat, sample_image_concat_2), dim=0)

        return sample, target_image


class TransformFixMatch(object):
    def __init__(self, ):
        self.weak = transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=456,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=456,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

