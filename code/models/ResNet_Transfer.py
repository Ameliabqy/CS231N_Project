import torch
import torch.nn as nn
import torchvision


class Resnet18_Transfer(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, num_classes=35):
        
        super(Resnet18_Transfer, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.layer4 = torchvision.models.resnet18(pretrained=False).layer4
        self.layer4=nn.Sequential(*self.layer4)

        
        
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
        
        x = self.layer4(x) # train the last block ourselves
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

    
class Resnet50_Transfer(nn.Module):
    
    
    def __init__(self, num_classes=35):
        
        super(Resnet50_Transfer, self).__init__()
        
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.layer4 = torchvision.models.resnet50(pretrained=False).layer4
        self.layer4=nn.Sequential(*self.layer4)
    
        
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
        
        x = self.layer4(x) # train last block ourselves
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_upsampled = nn.functional.upsample(logits_8s, size=input_spatial_dim, mode = 'bilinear', align_corners = True)
        
        return logits_upsampled