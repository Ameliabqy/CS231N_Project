import torch
import torch.nn as nn
import torchvision
    
class Resnet50_Deconv(nn.Module):
    
    
    def __init__(self, num_classes=35):
        
        super(Resnet50_Deconv, self).__init__()
              
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
            
#         self.resnet50 = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-1])
        temp_list = list(torchvision.models.resnet50(pretrained=True).children())[:-1]
        for layer in temp_list:
            layer.requires_grad = False

#         temp_list.append(list(torchvision.models.resnet50(pretrained=False).children())[-1])
        self.resnet50 = nn.Sequential(*temp_list)
#         self.resnet50.children().remove(self.resnet50.children()[-1])
        self.last_resnet_layer = torchvision.models.resnet50(pretrained=False).layer4
        self.last_resnet_layer = nn.Sequential(*self.last_resnet_layer)
            
#         for param in self.resnet50.parameters():
#                 param.requires_grad = False
                
                
        self.layer1 = nn.Conv2d(2048,
                                   num_classes,
                                   kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        
        self.layer2 = nn.Conv2d(1024,
                                   num_classes,
                                   kernel_size=3)
        self.relu2 = torch.nn.ReLU()
        
        self.layer3 = nn.Conv2d(512,
                                   num_classes,
                                   kernel_size=3)
        self.relu3 = torch.nn.ReLU()
        
        
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1)
        self.layer4 = nn.Conv2d(512,
                           num_classes,
                           kernel_size=3)
        self.relu4 = torch.nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1)
        self.layer5 = nn.Conv2d(512,
                           num_classes,
                           kernel_size=3)
        self.relu5 = torch.nn.ReLU()
        
        self.deconv3 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=8, padding=1)
        
        
    def forward(self, x):
        
        print("input: ", x.shape)
        x = self.resnet50(x)
        print("after resnet50: ", x.shape)
        x = self.last_resnet_layer(x)
        print("after last_resnet_layer: ", x.shape)
        x = self.layer1(x)
        print("layer1: ", x.shape)
        x = self.relu1(x)
        print("relu1: ", x.shape)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        
        x = self.deconv1(x)
        x = self.layer4(x)
        x = self.relu4(x)
        
        x = self.deconv2(x)
        x = self.layer5(x)
        x = self.relu5(x)
        
        x = self.deconv3(x)
        
        return x
        
        
#         input_spatial_dim = x.size()[2:]
        
#         x = self.resnet50.conv1(x)
#         x = self.resnet50.bn1(x)
        
#         x = self.resnet50.relu(x)
#         x = self.resnet50.maxpool(x)

#         x = self.resnet50.layer1(x)
        
#         x = self.resnet50.layer2(x)
#         logits_8s = self.score_8s(x)
        
#         x = self.resnet50.layer3(x)
#         logits_16s = self.score_16s(x)
        
#         x = self.layer4(x) # train last block ourselves
#         logits_32s = self.score_32s(x)
        
#         logits_16s_spatial_dim = logits_16s.size()[2:]
#         logits_8s_spatial_dim = logits_8s.size()[2:]
        
#         logits_16s += self.deconv1(logits_32s, output_size=logits_16s_spatial_dim)
        
#         logits_8s += self.deconv2(logits_16s, output_size=logits_8s_spatial_dim)
        
#         logits_upsampled = self.deconv3(logits_8s, output_size=input_spatial_dim)
        
#         return logits_upsampled