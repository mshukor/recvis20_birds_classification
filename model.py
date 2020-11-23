import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class EFFICIENT(nn.Module):
    def __init__(self, num_classes):
        super(EFFICIENT, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.model(x)
        # out = out.view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu(self.fc2(x))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(x)
        return out


 

#### https://github.com/dipuk0506/SpinalNet/blob/master/CIFAR-10/ResNet_default_and_SpinalFC_CIFAR10.py

class SpinalNet_FC(nn.Module):
    def __init__(self, num_classes=20):
        super(SpinalNet_ResNet, self).__init__()
        
                
        model_ft = models.wide_resnet101_2(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        half_in_size = round(num_ftrs/2)
        layer_width = 20 #Small for Resnet, large for VGG
        Num_class = num_classes

        self.fc_spinal_layer1 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(layer_width*4, Num_class),)
        
    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,half_in_size:2*half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,half_in_size:2*half_in_size], x3], dim=1))
        
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        
        x = self.fc_out(x)
        return x
 









