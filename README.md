## Fine Grained Image Classification

### Introduction
This repository includes the solution that achieves 85 % accuracy on  the public leaderboard of the private competition during the MVA Master.

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```
Also you need to clone and install detectron to generate the cropped images:
```bash

pip install --upgrade mxnet-cu101 gluoncv
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```
#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

#### General Pipeline

1. Run `preprocess_detectron.py` to generate cropped images using MaskRCNN trained on COCO dataset (this will help the model to focus on birds).
2. Train and Generate Attention images using `main_wsdan.py` and `preprocess_attention.py`. 
3. Run and experiment different approaches by running and changing the parameters of `main.py`.
4. You can test also some semi and self supervised approaches (including the SOTA FixMatch) by trying `train_fix_match.py`, `train_rotate.py`, `train_semi_ae.py` and `train_semi_self.py`.
5. Finally validate and evaluate your model using one of the provided scripts (some include ensembling evaluation).


#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Adaptation done by Gul Varol: https://github.com/gulvarol
