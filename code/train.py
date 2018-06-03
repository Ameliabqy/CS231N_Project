from cvpr import *
from DownUp import *
from ResNet18 import *
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

predict_im_path = "./predict_im/"

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device('cuda' if use_cuda else "cpu")
print(device)


# Create the CVPR dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
trainset = CVPR(hp,
    preload=hp.preload, transform=transforms.ToTensor(), train_sel = True)
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=hp.batch_size, shuffle=False, num_workers=1)
    
#     valset = CVPR(hp,
#         preload=True, transform=transforms.ToTensor(), train_sel = False
#     )
#     # Use the torch dataloader to iterate through the dataset
#     valset_loader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=1)

# data = np.asarray( trainset.labels[0], dtype="float32" )
# print(data.max())
# for t, (x, y) in enumerate(trainset_loader):
#     print(y.max())
#     print(x)
#     break

# print(np.where(data == 33000))

# transform = transforms.ToTensor()
# y = transform(trainset.labels[4])
# print(y.unique())
# print(y.shape)
# y = y.view(1,1,y.shape[1], y.shape[2])
# y = Crop(y, (1680, 1690, 330, 340))


# y1 = ConvertLabels(y)
# y2 = ReverseConvertLabels(y1)


def create_optimizer(model, hp):
    
    optimizer = None
    if hp.optimizer == "Adam":    
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay = hp.lr_decay)
    if hp.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=hp.learning_rate)
    if hp.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate, weight_decay=hp.lr_decay)
    if hp.optimizer == "RMSProp":
        optimizer = torch.optim.RMSProp(model.parameters(), lr=hp.learning_rate, weight_decay=hp.lr_decay, momentum=hp.momentum, eps=1e-10)
    
    return optimizer

def check_accuracy(loader, model, save_flag):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on validation set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
#         trainset_loader.update_iterator()
#         x, y = trainset_loader.get_images();
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=hp.dtype)
            y *= 255
            preds = model(x)
            if device == torch.device('cuda'):
                preds = preds.cuda()
            preds = ConvertOutputToLabels(preds)
            if device == torch.device('cuda'):
                y = y.cuda()
            y = y.squeeze()
    #             print(np.unique(np.asarray(preds)), np.unique(np.asarray(y)))
            plus_num_correct = (preds.type_as(y) == y).sum()
            plus_num_samples = y.numel()
            num_correct += plus_num_correct
            num_samples += plus_num_samples
            current_acc = float(plus_num_correct) / plus_num_samples
            if save_flag and current_acc * 100 > 89.8:
                im_np = np.asarray( preds, dtype="int8" )
                im = Image.fromarray(im_np[1, :, :].squeeze(), mode = "P")
                im.save(predict_im_path + str(t)+".png")
                im_label_np = np.asarray( y, dtype="int8" )
                im_label = Image.fromarray(im_label_np[1, :, :].squeeze(), mode = "P")
                im_label.save(predict_im_path + str(t)+"_label.png")
                del im_np, im, im_label_np, im_label
        print('in loop (%.2f)', current_acc * 100)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        
        
def train(model, create_optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    print('start training ')
    if hp.num_epochs:
        epochs = hp.num_epochs
    for e in range(epochs):
#         trainset.update_iterator()
#         x, y = trainset_loader.get_images();
        for t, (x, y) in enumerate(trainset_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=hp.dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=hp.dtype)
            y *= 255
            y, weights = ConvertCELabels(y)
            if device == torch.device('cuda'):
                y = y.cuda()
                weights = weights.cuda()
            scores = model(x)

            if hp.loss_type == "full":
                loss_func = F.torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_func(scores, y)
    #             print(scores)
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
                check_accuracy(trainset_loader, model, True)
                print()


# model = DownUp()
# model = Resnet18_8s()
model = Resnet50_8s()
train(model, create_optimizer)

