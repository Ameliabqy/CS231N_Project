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

# resnet18 = models.resnet18(pretrained=True)
# modules = list(resnet18.children())[:-1]
# modules.append(nn.Upsample(scale_factor=7, mode='bilinear', align_corners=True))
# modules.append(nn.Conv2d(512, 256, (3, 3), bias=True))
# modules.append(nn.ReLU())
# modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
# modules.append(nn.Conv2d(256, 256, (3, 3), bias=True))
# modules.append(nn.ReLU())
# modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
# modules.append(nn.Conv2d(256, 128, (3, 3), bias=True))
# modules.append(nn.ReLU())
# modules.append(nn.Conv2d(128, 128, (3, 3), bias=True))
# modules.append(nn.ReLU())
# modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
# modules.append(nn.Conv2d(128, 64, (3, 3), bias=True))
# modules.append(nn.ReLU())
# modules.append(nn.Conv2d(64, 64, (3, 3), bias=True))
# modules.append(nn.ReLU())
# modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
# modules.append(nn.Conv2d(64, 3, (7, 7), bias=True))
# modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
# resnet18=nn.Sequential(*modules)
# for p in resnet18.parameters():
#     p.requires_grad = False

class Resnet18(nn.Module):
    def __init__(self):
        
        super(Resnet18, self).__init__()
        
        # Load the pretrained weights
        resnet18 = models.resnet18(pretrained=True)
        #took out last block, avgpool, and linear layer of ResNet 18
        modules = list(resnet18.children())[0:7]
        # last block of Resnet 18
        modules += [nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        self.resnet18=nn.Sequential(*modules)
        self.upsample1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(512, 256, (3, 3), bias=True)
        self.relu1 = nn.ReLU()
        self.upsample2 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 128, (3, 3), bias=True)
        self.relu2 = nn.ReLU()
        self.upsample3 = nn.ConvTranspose2d(128, 128, 3, stride=4, padding=1)
        self.conv3 = nn.Conv2d(128, 64, (3, 3), bias=True)
        self.upsample4 = nn.ConvTranspose2d(64, 35, 3, stride=2, padding=1)
        for p in self.resnet18.parameters():
            p.requires_grad = True
                
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.upsample4(x)
        x = nn.functional.upsample(input = x, size = input_spatial_dim, mode = 'bilinear', align_corners = True)
        return x
    
    

