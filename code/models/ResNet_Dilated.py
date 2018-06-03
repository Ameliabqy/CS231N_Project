import torch
import torch.nn as nn
import torchvision


class Resnet18_Dilated(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, num_classes=35):
        
        super(Resnet18_Dilated, self).__init__()
        
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
                
        logits_16s_layer = nn.ConvTranspose2d(logits_32s.size()[1], logits_32s.size()[1], 3, stride=2, padding=1, dilation = (2,2))
        logits_16s += logits_16s_layer(logits_32s, output_size=logits_16s_spatial_dim)[:,:,1:-1,1:-1]
        
        logits_8s_layer = nn.ConvTranspose2d(logits_16s.size()[1], logits_16s.size()[1], 3, stride=2, padding=1, dilation = 2)
        logits_8s += logits_8s_layer(logits_16s, output_size=logits_8s_spatial_dim)[:,:,1:-1,1:-1]
        
        logits_upsampled_layer = nn.ConvTranspose2d(logits_8s.size()[1], logits_8s.size()[1], 3, stride=8, padding=1, dilation = 2)
        logits_upsampled = logits_upsampled_layer(logits_8s, output_size=input_spatial_dim)[:,:,1:-1,1:-1]
        
        return logits_upsampled

    
class Resnet50_Dilated(nn.Module):
    
    
    def __init__(self, num_classes=35):
        
        super(Resnet50_Dilated, self).__init__()
        
        
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
        
        logits_16s_layer = nn.ConvTranspose2d(logits_32s.size()[1], logits_32s.size()[1], 3, stride=2, padding=1, dilation = 2)
        logits_16s += logits_16s_layer(logits_32s, output_size=logits_16s_spatial_dim)
        
        logits_8s_layer = nn.ConvTranspose2d(logits_16s.size()[1], logits_16s.size()[1], 3, stride=2, padding=1, dilation = 2)
        logits_8s += logits_8s_layer(logits_16s, output_size=logits_8s_spatial_dim)
        
        logits_upsampled_layer = nn.ConvTranspose2d(logits_8s.size()[1], logits_8s.size()[1], 3, stride=8, padding=1, dilation = 4)
        logits_upsampled = logits_upsampled_layer(logits_8s, output_size=input_spatial_dim)
        
        return logits_upsampled