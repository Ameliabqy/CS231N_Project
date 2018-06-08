import torch
import torch.nn as nn
import torchvision

class BasicBlock(nn.Module):
    def __init__(self, planes, stride=1, dilation = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=dilation, dilation = (dilation, dilation), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=dilation, dilation = (dilation, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

    
class DRN_A(nn.Module):
    
    def __init__(self, num_classes=35):
        
        super(DRN_A, self).__init__()
        
        # layer and get the output stride of 8
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding = 3, bias=False), 
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                    BasicBlock(64))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 
                                    BasicBlock(128),
                                    BasicBlock(128))
        
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 
                                    BasicBlock(256, dilation = 2),
                                   BasicBlock(256, dilation = 2))
        
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    BasicBlock(512, dilation = 4),
                                   BasicBlock(512, dilation = 4))
        self.final = nn.Sequential(
            nn.Conv2d(512, 35, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(35, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(35, num_classes, kernel_size=1)
        )
        
       
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        
        x = self.final(x)
        
        
        logits_upsampled = nn.functional.upsample(x, size=input_spatial_dim, mode = 'bilinear', align_corners = True)
        return logits_upsampled
