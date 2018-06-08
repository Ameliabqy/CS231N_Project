from cvpr import *
from models.DownUp import * 
from models.ResNet import * 
from models.ResNet_Transfer import * 
from models.ResNet_Deconv import * 
from models.ResNet_Dilated import * 
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

train_accuracies = np.array([])
val_accuracies = np.array([])
losses = np.array([])

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device('cuda' if use_cuda else "cpu")
print(device)


# Create the CVPR dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
trainset = CVPR(hp,
    preload=False, transform=transforms.ToTensor(), train_sel = True)
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=hp.batch_size, shuffle=True, num_workers=1)

valset = CVPR(hp,
    preload=False, transform=transforms.ToTensor(), train_sel = False)
# Use the torch dataloader to iterate through the dataset
valset_loader = DataLoader(valset, batch_size=hp.batch_size, shuffle=True, num_workers=1)



        
        
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def create_optimizer(model, hp):
    
    optimizer = None
    if hp.optimizer == "Adam":    
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    if hp.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=hp.learning_rate)
    if hp.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay, momentum=hp.momentum, nesterov=hp.use_Nesterov)
    if hp.optimizer == "RMSProp":
        optimizer = torch.optim.RMSProp(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay, momentum=hp.momentum, eps=1e-10)
    
    return optimizer



def check_accuracy(loader, model, save_flag):
    global train_accuracies
    global val_accuracies
    if loader.dataset.train:
        print('')
        print('Checking accuracy on train set')
    else:
        print('')
        print('Checking accuracy on validation set')   
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
            im_np = np.asarray( preds, dtype="int8" )
            
            # Save images
            if save_flag and current_acc * 100 > 1: 
                if loader.dataset.train:
                    imstr = 'Pred'
                    xstr = 'Orig'
                else:
                    imstr = 'PredVal'
                    xstr = 'OrigVal'
                    
                torchvision.utils.save_image(x[1, :, :, :], xstr + str(t)+".png")
                
                im_np = np.asarray( preds, dtype="int8" )
                im = Image.fromarray(im_np[1, :, :].squeeze(), mode = "P")
                im.save(imstr + str(t)+".png")
                
                im_label_np = np.asarray( y, dtype="int8" )
                im_label = Image.fromarray(im_label_np[1, :, :].squeeze(), mode = "P")
                im_label.save(imstr + str(t)+"_label.png")
                
                del im_np, im, im_label_np, im_label
            print('Batch %d: %.2f' % (t, current_acc * 100))
            if t > 10:
                break
            
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        if loader.dataset.train:
            train_accuracies = np.append(train_accuracies, acc)
        else: 
            val_accuracies = np.append(val_accuracies, acc)
        
        
        
def train(model, create_optimizer, epochs=1):
    global losses
    global train_accuracies
    global val_accuracies
    optimizer = create_optimizer(model, hp)
    load_checkpoint('saved_model__Resnet50_Transfer__05AM_June_07.pt', model, optimizer)
    model.train()  # put model to training mode
    
    if hp.num_epochs:
        epochs = hp.num_epochs
    for e in range(epochs):
        
        print("")
        print("STARTING EPOCH %d" % e)
        print("")
        for t, (x, y) in enumerate(trainset_loader):
            x = x.to(device=device, dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=hp.dtype)
            y *= 255
            y[y == 255] = 0
            x = x.cuda()
            y = y.cuda()
            y = ConvertCELabels(y)
            if device == torch.device('cuda'):
                y = y.cuda()
                weights = hp.weights.cuda()
                scores = model(x).cuda()

            if hp.loss_type == "full":
                loss_func = F.torch.nn.CrossEntropyLoss(weight=weights, reduce=False)
                loss = loss_func(scores, y)
            
            mask = torch.ones(loss.size(), device=device, dtype=hp.dtype)
            mask *= 0.05 #good results with 0.05
            
            for n in range(y.shape[0]):
                indices = np.argwhere(y[n, :, :] > 0)
                if indices.numel() > 0:
                    if indices.shape[1] >= 2:
                        xmin = torch.min(indices[0, :]).item()
                        ymin = torch.min(indices[1, :]).item()
                        xmax = torch.max(indices[0, :]).item()
                        ymax = torch.max(indices[1, :]).item()
                        if xmin != xmax and ymin != ymax:
                            mask[n, xmin:xmax, ymin:ymax] = 1
                    
            N, H, W = loss.shape

            loss = torch.sum(torch.mul(loss, mask)) / (N * H * W)
            losses = np.append(losses, loss.item())

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % hp.print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(trainset_loader, model, True)
                check_accuracy(valset_loader, model, True)
                print()
                
            if t % hp.save_every == 0:
                date_string = datetime.datetime.now().strftime("%I%p_%B_%d")
                model_string = 'saved_model__' + hp.model_name + '__'+ date_string + '.pt'
                save_checkpoint(model_string, model, optimizer)
#                 torch.save(model, 'saved_model__' + hp.model_name + '__'+ date_string + '.pt')
                
        np.save('Losses3.npy', losses)
        np.save('Train_Accuracies3.npy', train_accuracies)
        np.save('Val_Accuracies3.npy', val_accuracies)


        
        
## Define models 

## barebone model
# model = DownUp()

## Resnet pretrained model with upsampling 
# model = Resnet18_8s() # learning rate: 
# model = Resnet50_8s() # learning rate:

# Resnet with upsampling using transfer learning 
if hp.model_name == 'Resnet18_Transfer':
    model = Resnet18_Transfer()
if hp.model_name == 'Resnet50_Transfer':
    model = Resnet50_Transfer()
    

## Deconvolution upsampling with transfer learning 
# model = Resnet18_Deconv() # learning rate:
# model = Resnet50_Deconv() # learning rate:

## Dilated deconvolution layers with transfer learning 
# model = Resnet18_Dilated() # learning rate:
# model = Resnet50_Dilated() # learning rate:
# model = torch.load('saved_model__Resnet50_Transfer__05AM_June_07.pt', model, )
# model = model.cuda()
model = torch.nn.DataParallel(model).cuda()
train(model, create_optimizer)

