import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image

class DownUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.pool1 = nn.MaxPool2d((4, 4), stride=(4, 4), padding = 1)
        
        self.conv2 = nn.Conv2d(10, 20, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.conv3 = nn.Conv2d(20, 10, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.conv4 = nn.Conv2d(10, 1, (3, 3), padding=(0,1), bias=True)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        
        
        
    def forward(self, x):
        scores = None
        # Downsample
        layer1 = self.conv1(x)
        layer2 = F.relu(layer1)
        layer3 = self.pool1(layer2)
        layer4 = self.conv2(layer3)
        layer5 = F.relu(layer4)
#         Down = F.relu(self.conv2(self.pool1(F.relu(self.conv1(x)))))
        
        layer6 = self.up1(layer5)
        layer7 = self.conv3(layer6)
        layer8 = F.relu(layer7)
        scores = self.conv4(layer8)
#         scores = self.conv4(F.relu(self.conv3(self.up1(layer5))))
        
        return scores


