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
import gc

import matplotlib.pyplot as plt

class_defines = np.array([0, 1, 17, 33, 34, 35, 36, 37, 38, 39, 40, 49, 50, 65, 66, 67, 81, 82, 83, 84, 85, 86, 97, 98, 99, 100, 113, 161, 162, 163, 164, 165, 166, 167, 168], dtype = np.int32)

class HyperParameters:
    """
    Object to hold all hyperparameters. Makes passing parameters between functions easier.
    Many of these might not be strictly necessary.
    
    """
    def __init__(self):
        # General params
        self.dtype = torch.float
        self.train_root = '../../../CS231N_Project/data/cropped_train_color'
        self.val_root = '../../../CS231N_Project/data/cropped_val_color'
        
        # Training params
        self.optimizer = "SGD" # options: SGD, RMSProp, Adam, Adagrad
        self.learning_rate = 5e-4 #9e-3 resnet18 SGD
        self.lr_decay = 0.9
        self.loss_type = "full"  # options: "fast", "full"
        self.momentum = 0.99
        self.use_Nesterov = True
        self.init_scale = 3.0
        self.num_epochs = 100  # Total data to train on = num_epochs*batch_size
        
        # Data loader params
        self.shuffle_data = True  # Currently doesn't do anything
        self.preload = False
        self.batch_size = 100
        self.num_files_to_load = self.num_epochs * self.batch_size
        
        self.num_classes = 20  # This value is probably wrong
        self.print_every = 3
        self.show_every = 5

        # Graph saving params
        self.save_model = True
        self.use_saved_model = False
    
hp = HyperParameters()

class CVPR(Dataset):
    """
    A customized data loader for CVPR.
    """
    def __init__(self, hp,
                 transform=None,
                 preload=False, train_sel = True):
        """ Intialize the CVPR dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        # Important - This tracks which batch we are on. Used for loading as we go (as opposed to preloading)
        self.batch_num = 0
        
        self.images = None
        self.labels = None
        self.filenames = []
        self.id_map = {}
        self.image_ids = []
        self.image_info = {0:{'source': 'none'}}
        self.source_class_ids = {'none':0}
        if train_sel:
            self.root = hp.train_root
            self.train = True
        else:
            self.root = hp.val_root
            self.train = False
        self.transform = transform
        self.num_classes = 35

        # read filenames
        filenames = glob.glob(osp.join(self.root, '*.jpg'))
        t = 0
        for fn in filenames:
            lbl = fn[:-4] + '_instanceIds.png'
            lbl = lbl.replace('color', 'label')
            self.filenames.append((fn, lbl)) # (filename, label) pair
            if train_sel:
                self.id_map[fn] = t
                t += 1
                if len(self.filenames) >= hp.num_files_to_load: 
                    break
            else:
                if len(self.filenames) >= hp.batch_size * 10: 
                    break
        self.labels = []
        self.images = []
        self._preload()
        
        self.len = len(self.filenames)
        self.image_ids = [0] * self.len
        
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        for image_fn, label in self.filenames:            
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
            # load labels
            label = Image.open(label)
            # avoid too many opened files bug
            self.labels.append(label.copy())
            label.close()

    def load_image(self, filename):
        """ Get a sample from the dataset
        """
        image = self.images[filename]
        image = np.asarray(image)
        # return image
        return image

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
    def __iter__(self):
        N, B = hp.num_files_to_load, hp.batch_size
        return iter((self.images[i:i+B], self.labels[i:i+B]) for i in range(0, N, B))
    
    def update_iterator(self):
        # If a batch has already been loaded free the last batch
        if len(self.images) != 0:
            for image in self.images:
                del image
            for label in self.labels:
                del label
        del self.images
        del self.labels
        self.images = []
        self.labels =[]
        gc.collect()
        i = self.batch_num*hp.batch_size
        train_filenames = self.filenames[i:i + hp.batch_size]
        for image_fn, label in train_filenames:
            image = Image.open(image_fn)
            self.images.append(self.normalize_image(np.asarray(image)))
            image.close()
            del image
            image = Image.open(label)
            self.labels.append((np.asarray(image)).astype(int))
            image.close()
            del image
        gc.collect()
        self.batch_num += 1
        
    def load_mask(self, image_id):
        y = np.asarray(self.labels[image_id], dtype = np.int32)
        mask = ConvertCELabels(y)
        mask = np.asarray(mask)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, class_defines
        

# takes tensor of size N x C x H x W, 
def ConvertLabels(labels):
    N, _, H, W = labels.shape
    C = 35
    converted_labels = torch.zeros([N, C, H, W], dtype=torch.int64)
    for i in range(C):
        mask = torch.eq(labels.type_as(class_defines), class_defines[i])
        converted_labels[:, i, :, :] = mask.view(N, H, W)
    return converted_labels
       
    
# converts output of forward pass to label format
# takes tensor of size N x C x H x W with last layer sigmoid for classes, gets the maximum category
# and converts that category to original labels
def ConvertOutputToLabels(output):
    N, C, H, W = output.shape
    output = torch.argmax(output, dim=1)
    converted_output = torch.zeros(output.shape, dtype=torch.int64)
    for c in range(C):
        converted_output[output == c] = class_defines.data[c]
    return converted_output
    
    
# makes labels for training with cross entropy loss
# takes tensor of size N x 1 x H x W with numbers for classes, changes those numbers to classification indices
def ConvertCELabels(label):
    H, W = label.shape
    C = 35
    mask = np.ones((H, W, C), dtype=bool)
    for c in range(C):
        mask_c = np.equal(label, class_defines[c])
        mask[:, :, c] = mask_c
    return mask
