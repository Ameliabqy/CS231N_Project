# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# import torchvision.models as models
# import glob
# import os.path as osp

# class Resnet18(nn.Module):
#     def __init__(self):
        
#         super(Resnet18, self).__init__()
        
#         # Load the pretrained weights
#         resnet18 = models.resnet18(pretrained=True)
#         #took out last block, avgpool, and linear layer of ResNet 18
#         modules = list(resnet18.children())[0:7]
#         # last block of Resnet 18
#         modules += [nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
#                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
#         self.resnet18=nn.Sequential(*modules)
#         self.upsample1 = nn.ConvTranspose2d(512, 512, 3, stride=4, padding = 2)
#         self.conv1 = nn.Conv2d(512, 256, (3, 3), padding = 1, bias=True)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(256, 256, (3, 3), padding = 1, bias=True)
#         self.relu2 = nn.ReLU()
#         self.upsample2 = nn.ConvTranspose2d(256, 256, 3, stride=2)
#         self.conv3 = nn.Conv2d(256, 128, (3, 3), padding = 1, bias=True)
#         self.relu3 = nn.ReLU()
#         self.conv4 = nn.Conv2d(128, 128, (3, 3), padding = 1, bias=True)
#         self.relu4 = nn.ReLU()
#         self.upsample3 = nn.ConvTranspose2d(128, 128, 3, stride=2)
#         self.conv5 = nn.Conv2d(128, 64, (3, 3), padding = 1, bias=True)
#         self.relu5 = nn.ReLU()
#         self.conv6 = nn.Conv2d(64, 64, (3, 3), padding = 1, bias=True)
#         self.relu6 = nn.ReLU()
#         self.upsample4 = nn.ConvTranspose2d(64, 35, 3, stride=2)
                
        
#     def forward(self, x):
        
#         input_spatial_dim = x.size()[2:]
        
#         x = self.resnet18(x)
#         x = self.upsample1(x)
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.upsample2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.upsample3(x)
#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.conv6(x)
#         x = self.relu6(x)        
#         x = self.upsample4(x)
#         x = nn.functional.upsample(input = x, size = input_spatial_dim, mode = 'bilinear', align_corners = True)
#         return x
    
    
import torch
import torch.nn as nn
import torchvision


class Resnet18_8s(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, num_classes=35):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        modules = list(self.resnet18.children())[0:-1]
        resnet18_32s=nn.Sequential(*modules)
        
        
        # Create a linear layer -- we don't need logits in this case
        resnet18_32s.fc = nn.Sequential()
        
        self.resnet18_32s = resnet18_32s
        
        self.score_32s = nn.Conv2d(512,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128,
                                   num_classes,
                                   kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        
        x = self.resnet18.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet18.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet18.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_upsampled = nn.functional.upsample(input = logits_8s,
                                                           size=input_spatial_dim, mode = 'bilinear', align_corners = True)
        
        return logits_upsampled

    
class Resnet50_8s(nn.Module):
    
    
    def __init__(self, num_classes=35):
        
        super(Resnet50_8s, self).__init__()
        
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        modules = list(self.resnet50.children())[0:-1]
        resnet50_32s=nn.Sequential(*modules)
        
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()
        
        self.resnet50_32s = resnet50_32s
        
        self.score_32s = nn.Conv2d(2048 ,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(1024,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(512,
                                   num_classes,
                                   kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        
        x = self.resnet50.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet50.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet50.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_upsampled = nn.functional.upsample(logits_8s, size=input_spatial_dim, mode = 'bilinear', align_corners = True)
        
        return logits_upsampled