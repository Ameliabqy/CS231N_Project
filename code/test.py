from cvpr_test import *
from models.DownUp import * 
from models.ResNet import * 
from models.ResNet_Transfer import * 
from models.ResNet_Deconv import * 
from models.DRN import *
from models.Unet import *
from models.ResNet_Pyramid import *
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
import datetime

CUDA_VISIBLE_DEVICES = 0,1,2,3

test_accuracies = np.array([])
index = 0

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device('cuda' if use_cuda else "cpu")
print(device)


# Create the CVPR dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
testset = CVPR_test(hp,
    preload=False, transform=transforms.ToTensor())
# Use the torch dataloader to iterate through the dataset
testset_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
    
def check_accuracy(loader, model):
    global test_accuracies
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=hp.dtype)
            y *= 255
            y[y == 255] = 0
            
            preds = model(x)
            if device == torch.device('cuda'):
                preds = preds.cuda()
                y = y.cuda()
            preds = ConvertOutputToLabels(preds)
            y = y.squeeze()
            
            plus_num_correct = (preds.type_as(y) == y).sum()
            plus_num_samples = y.numel()
            num_correct += plus_num_correct
            num_samples += plus_num_samples
            current_acc = float(plus_num_correct) / plus_num_samples
            # Save images
            print('Test set %i', t)
            if t < 50:
                torchvision.utils.save_image(x[0, :, :, :], "./predict_im/PredtestRGB" + str(t)+".png")
                im_np = np.asarray( preds, dtype="int8" )
                im = Image.fromarray(im_np[0, :, :].squeeze(), mode = "P")
                im.save("./predict_im/Predtest" + str(t)+".png")
                im_label_np = np.asarray( y, dtype="int8" )
                im_label = Image.fromarray(im_label_np, mode = "P")
                im_label.save("./predict_im/Predtest" + str(t)+"_label.png")
                del im_np, im, im_label_np, im_label
                print('Batch %d: %.2f' % (t, current_acc * 100))
            
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            test_accuracies = np.append(test_accuracies, acc)
            np.save('Test_Accuracies.npy', test_accuracies)



            
model1 = Resnet50_Transfer() # Alice
model1 = torch.nn.DataParallel(model1).cuda()
checkpoint = torch.load('Alice.pt')
model1.load_state_dict(checkpoint['state_dict'])

    
# model2 = Resnet101_Pyramid() # Amelia
# model2 = torch.nn.DataParallel(model2).cuda()
# checkpoint = torch.load('SPP-Unet.pt')
# model2.load_state_dict(checkpoint['state_dict'])

# model3 = torch.load('Aristos.pt') # Aristos

check_accuracy(testset_loader, model1)
