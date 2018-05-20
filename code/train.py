from cvpr import *
from DownUp import *
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

hp = HyperParameters()

# Create the CVPR dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
if hp.preload:
    trainset = CVPR(hp,
        preload=True, transform=transforms.ToTensor(),
    )
    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

# # load the testset
# testset = CVPR(
#     root='mnist_png/testing',
#     preload=True, transform=transforms.ToTensor(),
# )
# # Use the torch dataloader to iterate through the dataset
# testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device(cuda if use_cuda else "cpu")
print(device)

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


def train(model, create_optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if hp.num_epochs:
        epochs = hp.num_epochs
    for e in range(epochs):
        for t, (x, y) in enumerate(trainset_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=hp.dtype)

            scores = model(x)
            
            if hp.loss_type == "fast":
                loss = F.mse_loss(scores, y)
                
            optimizer = create_optimizer(model, hp)

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
#                 check_accuracy_part34(loader_val, model)
                print()


learning_rate = 1e-2
model = DownUp()
train(model, create_optimizer)