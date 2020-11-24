import os
from matplotlib import pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm 
from PIL import Image

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--dir', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--num-datasets', type=int, default=2, metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")


args = parser.parse_args()
selected_classes = ['030.Fish_Crow', '009.Brewer_Blackbird', '029.American_Crow', '011.Rusty_Blackbird', '031.Black_billed_Cuckoo']

input_folder = args.dir
number_of_classes = len(list(os.listdir(input_folder + '/train_images')))
number_of_datasets = args.num_datasets
print("number of classes = ", number_of_classes)
for data_folder in list(os.listdir(input_folder)): # Iterate over train, val and test
  directory = input_folder+'/'+data_folder
  print("\n wotking on :", data_folder)
  folder_id = 0
  idx = -1
  for folder in list(os.listdir(directory)): # Iterate over classes of birds
    # if folder_id % (number_of_classes/number_of_datasets) == 0 and idx < (number_of_datasets -1):
    #   idx += 1
    if folder in selected_classes:
      idx = 0
    else:
      continue
    print(idx)
    output_folder = input_folder + '_' + str(idx)
    os.makedirs(output_folder, exist_ok = True)
    os.makedirs(output_folder +'/'+data_folder+'/'+folder, exist_ok = True)
    class_folder = directory + '/' + folder

    for f in tqdm(os.listdir(class_folder)):
      if 'jpg' in f:
        path = class_folder + '/' + f
        img = np.array(Image.open(path))
        plt.imsave(path.replace(input_folder, output_folder), np.array(img), dpi=1000)
    folder_id += 1

plt.close()

