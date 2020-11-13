from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm

dirName = 'bird_dataset/train_images'

dirName_crop_pre = 'bird_dataset_cropped'
dirName_mask_pre = 'bird_dataset_masked'

dirName_crop = dirName_crop_pre + '/train_images'
dirName_mask = dirName_mask_pre + '/train_images'

try:
  os.mkdir(dirName_crop_pre)
  os.mkdir(dirName_mask_pre)

  os.mkdir(dirName_crop)
  os.mkdir(dirName_mask)
except FileExistsError:
  pass


net = model_zoo.get_model('mask_rcnn_fpn_resnet101_v1d_coco', pretrained=True)
kernel = np.ones((43, 43), 'uint8')

listOfFile = os.listdir(dirName)
completeFileList = list()
for file in tqdm(listOfFile):
  completePath = os.path.join(dirName, file)
  image_paths = os.listdir(completePath)

  try:
    os.mkdir(os.path.join(dirName_crop, file))
  except:
    pass
  try:
    os.mkdir(os.path.join(dirName_mask, file))
  except:
    pass

  for image in tqdm(image_paths):
    img_path = os.path.join(completePath, image)

    x, orig_img = data.transforms.presets.rcnn.load_test(img_path)

    _, scores, bboxes, masks = net(x)

    try:
      bboxes = bboxes[0].asnumpy()
      masks = masks[0].asnumpy()
      scores = scores[0].asnumpy()

      width, height = orig_img.shape[1], orig_img.shape[0]
      masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)

      msk = masks[0]
      
      dilate_img = cv2.dilate(msk, kernel, iterations=1)
      masked = cv2.bitwise_and(orig_img, orig_img,mask = dilate_img)

      left, top , right, bottom  = bboxes[0]

      masked = Image.fromarray(masked)
      cropped_masked = masked.crop( ( left, top, right, bottom) )  # size: 45, 45

      img = Image.fromarray(orig_img)

      cropped = img.crop( ( left, top, right, bottom) )  # size: 45, 45

      score = str(int(scores[0].tolist()[0]*100))

      full_path_crop = img_path.replace("bird_dataset", dirName_crop_pre)
      full_path_mask = img_path.replace("bird_dataset", dirName_mask_pre)


      prefix = ('/').join(full_path_crop.split("/")[:-1])
      name = full_path_crop.split("/")[-1].split(".")[0]
      full_path_crop = prefix +'/' + name + "_" + score + "_crop.jpg"

      prefix = ('/').join(full_path_mask.split("/")[:-1])
      name = full_path_mask.split("/")[-1].split(".")[0]
      full_path_mask = prefix +'/' + name + "_" + score + "_mask.jpg"

      try:
          cropped.save(full_path_crop, "JPEG", quality=80, optimize=True, progressive=True)
      except IOError:
          PIL.ImageFile.MAXBLOCK = cropped.size[0] * cropped.size[1]
          cropped.save(full_path_crop, "JPEG", quality=80, optimize=True, progressive=True)
      try:
          cropped_masked.save(full_path_mask, "JPEG", quality=80, optimize=True, progressive=True)
      except IOError:
          PIL.ImageFile.MAXBLOCK = cropped_masked.size[0] * cropped_masked.size[1]
          cropped_masked.save(full_path_mask, "JPEG", quality=80, optimize=True, progressive=True)
    except:
      print("error in : ", img_path)
      continue

