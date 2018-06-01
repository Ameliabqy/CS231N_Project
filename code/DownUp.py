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
        self.conv1 = nn.Conv2d(3, 5, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding = 0)
        
        self.conv2 = nn.Conv2d(5, 12, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.conv3 = nn.Conv2d(12, 16, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding = 0)
        
        self.conv4 = nn.Conv2d(16, 24, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.conv5 = nn.Conv2d(24, 32, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2), padding = 0)
        
        self.conv6 = nn.Conv2d(32, 64, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv7 = nn.Conv2d(64, 64, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        
        
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv8 = nn.Conv2d(64, 64, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.constant_(self.conv8.bias, 0)
        
        self.conv9 = nn.Conv2d(64, 32, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.constant_(self.conv9.bias, 0)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv10 = nn.Conv2d(32, 24, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv10.weight)
        nn.init.constant_(self.conv10.bias, 0)
        
        self.conv11 = nn.Conv2d(24, 16, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv11.weight)
        nn.init.constant_(self.conv11.bias, 0)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv12 = nn.Conv2d(16, 8, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv12.weight)
        nn.init.constant_(self.conv12.bias, 0)
        
        self.conv13 = nn.Conv2d(8, 35, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv13.weight)
        nn.init.constant_(self.conv13.bias, 0)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        
        
        
        
        
    def forward(self, x):
        scores = None
        
        # Downsample
        layer1 = F.relu(self.conv1(x))
        layer2 = self.pool1(layer1)
        layer3 = F.relu(self.conv3(F.relu(self.conv2(layer2))))
        layer4 = self.pool2(layer3)
        layer5 = F.relu(self.conv5(F.relu(self.conv4(layer4))))
        layer6 = self.pool3(layer5)
        layer7 = F.relu(self.conv7(F.relu(self.conv6(layer6))))
        
        # Upsample
        layer8 = self.up1(layer7)
        layer9 = F.relu(self.conv9(F.relu(self.conv8(layer8))))
        layer10 = self.up2(layer9)
        layer11 = F.relu(self.conv11(F.relu(self.conv10(layer10))))
        layer12 = self.up3(layer11)
        scores = self.softmax(self.conv13(F.relu(self.conv12(layer12))))


# used before image cropping i think

#         Down1 = (self.pool2(F.relu(self.conv3(F.relu(self.conv2(self.pool1(F.relu(self.conv1(x)))))))))
#         Down2 = F.relu(self.conv7(F.relu(self.conv6(self.pool3(self.conv5(F.relu(self.conv4(Down1))))))))
        
#         Up1 = F.relu(self.conv10(self.up2((F.relu(self.conv9(F.relu(self.conv8(self.up1(Down2)))))))))
#         scores = F.relu(self.conv13(F.relu(self.conv12(self.up3((F.relu(self.conv11(Up1))))))))
        
        return scores


