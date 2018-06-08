import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
    
class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out

    
class Resnet101_Pyramid(nn.Module):
    
    def __init__(self, num_classes=35):
        
        super(Resnet101_Pyramid, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
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
        
        x = self.resnet101.conv1(x)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        
        x = self.resnet101.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet101.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet101.layer4(x)
        logits_32s = self.score_32s(x)
        
        x = self.ppm(x)
        x = self.final(x)
                
        logits_32s_spatial_dim = logits_32s.size()[2:]
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
        
        logits_32s += nn.functional.upsample(x,
                                        size=logits_32s_spatial_dim, mode = 'bilinear', align_corners = True)
                
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim, mode = 'bilinear', align_corners = True)
        
        logits_upsampled = nn.functional.upsample(logits_8s, size=input_spatial_dim, mode = 'bilinear', align_corners = True)
                
        
        return logits_upsampled
