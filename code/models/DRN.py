import torch
import torch.nn as nn
import torchvision

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, dilation = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False, dilation = dilation)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
                                    BasicBlock(64),
                                    BasicBlock(64))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 
                                    BasicBlock(128),
                                    BasicBlock(128))
        
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 
                                    BasicBlock(256, 256, dilation = 2),
                                    BasicBlock(256, 256, dilation = 2))
        
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    BasicBlock(256, 512, dilation = 4),
                                    BasicBlock(512, 512, dilation = 4))
        
        
        self.score_32s = nn.Conv2d(512,
                                   num_classes,
                                   kernel_size=1).cuda()
        
        self.score_16s = nn.Conv2d(256,
                                   num_classes,
                                   kernel_size=1).cuda()
        
        self.score_8s = nn.Conv2d(128,
                                   num_classes,
                                   kernel_size=1)
        
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1, dilation = 2)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1, dilation = 2)
        self.deconv3 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=8, padding=1, dilation = 2)
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += self.deconv1(logits_32s, output_size=logits_16s_spatial_dim)[:,:,1:-1,1:-1]
        
        logits_8s += self.deconv2(logits_16s, output_size=logits_8s_spatial_dim)[:,:,1:-1,1:-1]
        
        logits_upsampled = self.deconv3(logits_8s, output_size=input_spatial_dim)[:,:,1:-1,1:-1]
        
        return logits_upsampled
