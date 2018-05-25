import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import glob
import os.path as osp

# upsampling problem with dimension 

resnet18 = models.resnet18(pretrained=True)
modules = list(resnet18.children())[:-1]
modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
modules.append(nn.Conv2d(512, 256, (3, 3), padding = 1, bias=True))
modules.append(nn.ReLU())
modules.append(nn.Conv2d(256, 256, (3, 3), padding = 1, bias=True))
modules.append(nn.ReLU())
modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
modules.append(nn.Conv2d(256, 128, (3, 3), padding = 1, bias=True))
modules.append(nn.ReLU())
modules.append(nn.Conv2d(128, 128, (3, 3), padding = 1, bias=True))
modules.append(nn.ReLU())
modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
modules.append(nn.Conv2d(128, 64, (3, 3), padding = 1, bias=True))
modules.append(nn.ReLU())
modules.append(nn.Conv2d(64, 64, (3, 3), padding = 1, bias=True))
modules.append(nn.ReLU())
modules.append(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
modules.append(nn.Conv2d(64, 3, (7, 7), padding = 3, bias=True))
resnet18=nn.Sequential(*modules)

for p in resnet18.parameters():
    p.requires_grad = False
    

