from cvpr import *
from DownUp import *
from ResNet50DownUp import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image

CUDA_VISIBLE_DEVICES=0,1

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device('cuda' if use_cuda else "cpu")
print(device)

hp = HyperParameters()

# Create the CVPR dataset. 
trainset = CVPR(hp, preload=False, transform=ToByteTensor(), train_sel = True)
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=hp.batch_size, shuffle=True, num_workers=1)
    
#     valset = CVPR(hp,
#         preload=True, transform=transforms.ToTensor(), train_sel = False
#     )
#     # Use the torch dataloader to iterate through the dataset
#     valset_loader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=1)




def create_optimizer(model, hp):
    
    optimizer = None
    if hp.optimizer == "Adam":    
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    if hp.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=hp.learning_rate)
    if hp.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate, weight_decay=hp.lr_decay)
    if hp.optimizer == "RMSProp":
        optimizer = torch.optim.RMSProp(model.parameters(), lr=hp.learning_rate, weight_decay=hp.lr_decay, momentum=hp.momentum, eps=1e-10)
    
    return optimizer

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on validation set')   
    num_samples = 0
    num_correct = 0
    num_pred_zeros = 0
    num_zeros = 0
    index = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            y[y == 255] = 0
            x = x.to(dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(dtype=hp.dtype)
            x = x.cuda()
            y = y.cuda()
            
            preds = model(x)
            preds.cuda()
            preds = ConvertOutputToLabels(preds)
            N, H, W = preds.shape
            
#             print(preds)
#             print(torch.unique(y.cpu()))
       
#             torchvision.utils.save_image(preds.view(N, 1, H, W), filename="Preds.png")
#             torchvision.utils.save_image(y.view(N, 1, H, W), filename="Truth.png")
            im_np = np.asarray( preds, dtype="int8" )
            im = Image.fromarray(im_np[1, :, :].squeeze(), mode = "P")
            im.save("Pred" + str(t)+".png")
            im_label_np = np.asarray( y, dtype="int8" )
            im_label = Image.fromarray(im_label_np[1, :, :].squeeze(), mode = "P")
            im_label.save("Pred" + str(t)+"_label.png")
            del im_np, im, im_label_np, im_label
            num_correct += torch.eq(preds.type_as(y), y.squeeze()).sum()
            num_pred_zeros += torch.eq(preds.type_as(y), 0).sum()
            num_zeros += torch.eq(y, 0).sum()
#             print(num_correct)
#             print(num_samples)
            num_samples += y.numel()
            index += 1
            if index > 3:
                break
            print('in loop')
        pred_zeros = float(num_pred_zeros) / num_samples
        zeros = float(num_zeros) / num_samples
        acc = float(num_correct) / num_samples
        print('Percentage of zeros: %.2f' % (100 * zeros))
        print('Percentage of predicted zeros: %.2f' % (100 * pred_zeros))
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        
        
def train(model, create_optimizer, epochs=1):
    print('start training ')
    optimizer = create_optimizer(model, hp)
    model.train()  # put model to training mode
    
    if hp.num_epochs:
        epochs = hp.num_epochs
        
    for e in range(epochs):
        for t, (x, y) in enumerate(trainset_loader):
            y[y == 255] = 0
            x = x.to(dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(dtype=hp.dtype)
            
            x = x.cuda()
            y = y.cuda()
#             print("here")
            
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()
            
            y, weights = ConvertCELabels(y)
            y = y.cuda()
            weights = weights.cuda()
            scores = model(x)
            scores = scores.cuda()
            
            if hp.loss_type == "full":
                loss_func = F.torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_func(scores, y)
            
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % hp.print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(trainset_loader, model)
                print()
                break

model = DownUp()
# model = ResNet18()
model =  torch.nn.DataParallel(model).cuda()
train(model, create_optimizer)

