import zipfile
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

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
# class DoubleChannels(torch.utils.data.Dataset):
#   def __init__(self, dataset_path, concat_dataset_path, transform=None):

#       self.dataset_path = dataset_path
#       self.concat_dataset_path = concat_dataset_path

#       classes, class_to_idx = self._find_classes(dataset_path)

#       samples_images = torchvision.datasets.folder.make_dataset(dataset_path, class_to_idx, extensions, is_valid_file)
#       if len(samples_images) == 0:
#           msg = "Found 0 files in subfolders of: {}\n".format(dataset_path)
#           if extensions is not None:
#               msg += "Supported extensions are: {}".format(",".join(extensions))
#           raise RuntimeError(msg)

#       samples_image_concat = make_dataset(concat_dataset_path, class_to_idx, extensions, is_valid_file)
#       if len(samples_image_concat) == 0:
#           msg = "Found 0 files in subfolders of: {}\n".format(concat_dataset_path)
#           if extensions is not None:
#               msg += "Supported extensions are: {}".format(",".join(extensions))
#           raise RuntimeError(msg)

#       self.loader = loader
#       self.extensions = extensions

#       self.classes = classes
#       self.class_to_idx = class_to_idx

#       self.samples_images = samples_images
#       self.samples_image_concat = samples
#       self.transform = transform
#       self.targets_images = [s[1] for s in samples_images]
#       self.samples_image_concat = [s[1] for s in samples_images_concat]


#     def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
#         """
#         Finds the class folders in a dataset.

#         Args:
#             dir (string): Root directory path.

#         Returns:
#             tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

#         Ensures:
#             No class is a subdirectory of another.
#         """
#         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#         classes.sort()
#         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#         return classes, class_to_idx

#   def __getitem__(self, index):

#         path_image, target_image = self.samples_images[index]
#         path_image_concat = path_image.replace(self.dataset_path, self.concat_dataset_path)
#         # path_image_concat, target_image_concat = self.samples_images_concat[index]

#         sample_image = np.array(Image.open(path_image))
#         sample_image_concat = np.array(Image.open(path_image_concat))

#         if self.transform is not None:
#             sample_image = self.transform(sample_image)
#         sample_image_concat = torch.from_numpy(sample_image_concat)
#         sample = torch.cat((sample_image, sample_image_concat), dim=0)

#         return sample, target

