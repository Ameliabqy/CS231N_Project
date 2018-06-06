
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


class HyperParameters:
    """
    Object to hold all hyperparameters. Makes passing parameters between functions easier.
    Many of these might not be strictly necessary.
    
    """
    def __init__(self):
        # General params
        self.dtype = torch.float
        self.train_root = '../data/cvpr-2018-autonomous-driving/cropped_train_color'
        self.val_root = '../data/cvpr-2018-autonomous-driving/cropped_val_color'
        
        # Training params
        self.optimizer = "Adam" # options: SGD, RMSProp, Adam, Adagrad
        self.learning_rate = 4e-6 #9e-3 resnet18 SGD
        self.weight_decay = 1e-5
        self.lr_decay = 0.99
        self.loss_type = "full"  # options: "fast", "full"
        self.momentum = 0.7
        self.use_Nesterov = True
        self.init_scale = 3.0
        self.num_epochs = 500  
        self.num_minibatches = 2 # Total data to train on = num_minibatches*batch_size
        
        self.class_defines = torch.tensor([0, 1, 17, 33, 34, 35, 36, 37, 38, 39, 40, 49, 50, 65, 66, 67, 81, 82, 83, 84, 85, 86, 97, 98, 99, 100, 113, 161, 162, 163, 164, 165, 166, 167, 168], dtype = torch.int64)
        self.weights = torch.ones(self.class_defines.shape, dtype=torch.float32)
        
        # Data loader params
        self.shuffle_data = True  # Currently doesn't do anything
        self.preload = False
        self.batch_size = 25
        self.num_files_to_load = self.num_minibatches * self.batch_size
        
        self.num_classes = 35  
        self.print_every = 1
        self.show_every = 5
        

        # Graph saving params
        self.save_model = True
        self.save_every = 10
        self.model_name = 'Resnet50_Transfer'
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
            if train_sel:
                if len(self.filenames) >= hp.num_files_to_load: 
                    break
            else:
                if len(self.filenames) >= hp.batch_size * 5: 
                    break
#         self.labels = []
#         self.images = []
        # if preload dataset into memory
        if hp.preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
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
            self.labels.append((np.asarray(image)).astype(int))
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


        
        
# class ToByteTensor(object):
#     """Convert a ``PIL.Image`` to tensor.

#     Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#     """

#     def __call__(self, pic):
#         """
#         Args:
#             pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

#         Returns:
#             Tensor: Converted image.
#         """

#         # handle PIL Image

#         img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#         # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#         nchannel = len(pic.mode)
        
#         img = img.view(pic.size[1], pic.size[0], nchannel)
#         # put it from HWC to CHW format
#         # yikes, this transpose takes 80% of the loading time/CPU
#         img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         return img        
        
        
# # takes tensor of size N x C x H x W, 
# def ConvertLabels(labels):
#     N, _, H, W = labels.shape
#     C = 35
#     converted_labels = torch.zeros([N, C, H, W], dtype=torch.int64)
#     for i in range(C):
#         mask = torch.eq(labels.type_as(hp.class_defines), hp.class_defines[i])
#         converted_labels[:, i, :, :] = mask.view(N, H, W)
#     return converted_labels
       
    
# converts output of forward pass to label format
# takes tensor of size N x C x H x W with last layer sigmoid for classes, gets the maximum category
# and converts that category to original labels
def ConvertOutputToLabels(output):
    N, C, H, W = output.shape
    output = torch.argmax(output, dim=1)
    converted_output = torch.zeros(output.shape, dtype=torch.int64)
    for c in range(C):
        converted_output[output == c] = hp.class_defines.data[c]
    return converted_output
    
    
# makes labels for training with cross entropy loss
# takes tensor of size N x 1 x H x W with numbers for classes, changes those numbers to classification indices
def ConvertCELabels(labels):
    N, _, H, W = labels.shape
    C = 35
#     print(labels.max())
    converted_labels = torch.zeros([N, H, W], dtype=torch.int64)
    for c in range(C):
        mask = torch.eq(labels.type_as(hp.class_defines), hp.class_defines[c]).type_as(hp.class_defines)
        if c == 0:
            hp.weights[0] = 0.1*(1 - float(mask.sum())/N/H/W)
#             print(hp.weights[0])
        converted_labels += mask.view(N, H, W) * c
    return converted_labels
        
 
