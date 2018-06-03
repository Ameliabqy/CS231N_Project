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
        
        self.conv2 = nn.Conv2d(5, 12, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding = 0)
        
        self.conv3 = nn.Conv2d(12, 36, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)

        
        self.conv4 = nn.Conv2d(36, 64, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.conv5 = nn.Conv2d(64, 96, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding = 0)
        
        self.conv6 = nn.Conv2d(96, 120, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv7 = nn.Conv2d(120, 156, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.conv8 = nn.Conv2d(156, 172, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.constant_(self.conv8.bias, 0)
        
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2), padding = 0)
        
        self.conv9 = nn.Conv2d(172, 196, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.constant_(self.conv9.bias, 0)
        
        self.conv10 = nn.Conv2d(196, 212, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv10.weight)
        nn.init.constant_(self.conv10.bias, 0)
        
        self.conv11 = nn.Conv2d(212, 240, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv11.weight)
        nn.init.constant_(self.conv11.bias, 0)
        
        
        
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv12 = nn.Conv2d(240, 212, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv12.weight)
        nn.init.constant_(self.conv12.bias, 0)
        
        self.conv13 = nn.Conv2d(212, 196, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv13.weight)
        nn.init.constant_(self.conv13.bias, 0)
        
        self.conv14 = nn.Conv2d(196, 172, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv14.weight)
        nn.init.constant_(self.conv14.bias, 0)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv15 = nn.Conv2d(172, 156, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv15.weight)
        nn.init.constant_(self.conv15.bias, 0)
        
        self.conv16 = nn.Conv2d(156, 120, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv16.weight)
        nn.init.constant_(self.conv16.bias, 0)
        
        self.conv17 = nn.Conv2d(120, 96, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv17.weight)
        nn.init.constant_(self.conv17.bias, 0)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv18 = nn.Conv2d(96, 64, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv18.weight)
        nn.init.constant_(self.conv18.bias, 0)
        
        self.conv19 = nn.Conv2d(64, 36, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv19.weight)
        nn.init.constant_(self.conv19.bias, 0)
        
        self.conv20 = nn.Conv2d(36, 35, (3, 3), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv20.weight)
        nn.init.constant_(self.conv20.bias, 0)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        
        
        
        
        
    def forward(self, x):
        scores = None
        
        # Downsample
        layer1 = F.relu(self.conv2(F.relu(self.conv1(x))))
        layer2 = self.pool1(layer1)
        layer3 = F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(layer2))))))
        layer4 = self.pool2(layer3)
        layer5 = F.relu(self.conv8(F.relu(self.conv7(F.relu(self.conv6(layer4))))))
        layer6 = self.pool3(layer5)
        layer7 = F.relu(self.conv11(F.relu(self.conv10(F.relu(self.conv9(layer6))))))
        
        # Upsample
        layer8 = self.up1(layer7)
        layer9 = F.relu(self.conv14(F.relu(self.conv13(F.relu(self.conv12(layer8))))))
        layer10 = self.up2(layer9)
        layer11 = F.relu(self.conv17(F.relu(self.conv16(F.relu(self.conv15(layer10))))))
        layer12 = self.up3(layer11)
        scores = self.softmax(F.relu(self.conv20(self.conv19(F.relu(self.conv18(layer12))))))


# used before image cropping i think

#         Down1 = (self.pool2(F.relu(self.conv3(F.relu(self.conv2(self.pool1(F.relu(self.conv1(x)))))))))
#         Down2 = F.relu(self.conv7(F.relu(self.conv6(self.pool3(self.conv5(F.relu(self.conv4(Down1))))))))
        
#         Up1 = F.relu(self.conv10(self.up2((F.relu(self.conv9(F.relu(self.conv8(self.up1(Down2)))))))))
#         scores = F.relu(self.conv13(F.relu(self.conv12(self.up3((F.relu(self.conv11(Up1))))))))
        
        return scores


