
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

import matplotlib.pyplot as plt

class_defines = torch.tensor([33, 34, 35, 36, 37, 38, 39, 40, 0, 1, 17, 161, 162, 163, 164, 165, 166, 167, 168, 49, 50, 65, 66, 67, 81, 82, 83, 84, 85, 86, 97, 98, 99, 100, 113], dtype = torch.int64)

class HyperParameters:
    """
    Object to hold all hyperparameters. Makes passing parameters between functions easier.
    Many of these might not be strictly necessary.
    
    """
    def __init__(self):
        # General params
        self.dtype = torch.float32
        self.train_root = '../data/cvpr-2018-autonomous-driving/cropped_train_color'
        self.val_root = '../data/cvpr-2018-autonomous-driving/cropped_val_color'
        self.device = '/cpu:0'
        
        # Training params
        self.optimizer = "Adam" # options: SGD, RMSProp, Adam, Adagrad
        self.learning_rate = 1e-5
        self.lr_decay = 0.99
        self.loss_type = "full"  # options: "fast", "full"
        self.momentum = 0.9
        self.use_Nesterov = True
        self.init_scale = 3.0
        self.num_epochs = 10  # Total data to train on = num_epochs*batch_size
        
        # Data loader params
        self.shuffle_data = True  # Currently doesn't do anything
        self.preload = True
        self.batch_size = 50
        self.num_files_to_load = self.num_epochs * self.batch_size
        
        self.num_classes = 20  # This value is probably wrong
        self.print_every = 3

        # Graph saving params
        self.save_model = True
        self.use_saved_model = False
    

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
        if train_sel:
            self.root = hp.train_root
            self.train = True
        else:
            self.root = hp.val_root
            self.train = False
        self.transform = transform

        # read filenames
        filenames = glob.glob(osp.join(self.root, '*.jpg'))
        for fn in filenames:
            lbl = fn[:-4] + '_instanceIds.png'
            lbl = lbl.replace('color', 'label')
            self.filenames.append((fn, lbl)) # (filename, label) pair
            if len(self.filenames) >= hp.num_files_to_load: 
                break
                
        # if preload dataset into memory
        if hp.preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
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

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)
            label = Image.open(label)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        # return image and label
        return image, label

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
            self.labels.append((np.asarray(image)/1000).astype(int))
            image.close()
            del image
        gc.collect()
        self.batch_num += 1
        
    def get_images(self):
        return self.images, self.labels

    def normalize_image(self, image):
        # Normalize the data: subtract the mean pixel and divide by std
        mean_pixel = image.mean(keepdims=True)
        std_pixel = image.std(keepdims=True)
        image = (image - mean_pixel) / std_pixel
        return image
    
    def reset_data_set(self):
        self.batch_num = 0
        np.random.shuffle(self.bfilenames)
        del self.images
        del self.labels
        gc.collect()
        self.images = []
        self.labels = []

def Crop(x, lim_indices):
    hmin, hmax, wmin, wmax = lim_indices
    x.data = x.data[:,:,hmin : hmax, wmin : wmax]
    return x

def ConvertLabels(labels):
    N = labels.shape[0]
    H = labels.shape[2]
    W = labels.shape[3]
    converted_labels = torch.zeros([N, 35, H*W], dtype=torch.int64)
    flatten_labels = labels.contiguous().view(N, -1)
    choose_class = 3
    for i in range(0, choose_class):
        flatten_labels.data /= 1000
        converted_labels[:, i, :] = flatten_labels.type_as(class_defines) == class_defines[i]
    converted_labels = converted_labels[:,0:choose_class,:].view(N, choose_class, H, W).to(torch.float32)
    return converted_labels
        
def ReverseConvertLabels(labels):
    N, C, H, W = labels.shape
    converted_labels = torch.zeros([N, 1, H, W], dtype=torch.int64)
    for i in range(C):
        converted_labels[:, 0, :, :] += labels[:, i, :, :].type_as(class_defines) * class_defines[i] * 1000
    unlabeled = converted_labels == 0
    converted_labels += unlabeled.type_as(converted_labels) * 255
    return converted_labels
        
        