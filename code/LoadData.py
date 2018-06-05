from cvpr_mask_rcnn import *
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


# Create the CVPR dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
trainset = CVPR(hp,
    preload=True, transform=transforms.ToTensor(), train_sel = True)

# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=hp.batch_size, shuffle=False, num_workers=1)

valset = CVPR(hp,
    preload=True, transform=transforms.ToTensor(), train_sel = False)
# Use the torch dataloader to iterate through the dataset
valset_loader = DataLoader(valset, batch_size=hp.batch_size, shuffle=False, num_workers=1)