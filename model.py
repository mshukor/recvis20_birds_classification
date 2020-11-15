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