# Split_data.py
# written by Alice Li
# 5/15/18
# 
# This splits our training data into 80% train, 10% val, 10% test

import numpy as np 
from numpy import random
np.random.seed(5)

import os
import os.path as osp
import glob
import subprocess as sp

root = '../data/'
ret = sp.call(['test', '-d', osp.join(root, 'val_color')])

if not ret:
	print ('Already split data, exiting:')
	quit()

print ("Creating folders")
sp.call(['mkdir', osp.join(root, 'val_color')])
sp.call(['mkdir', osp.join(root, 'val_label')])
sp.call(['mkdir', osp.join(root, 'test_color')])
sp.call(['mkdir', osp.join(root, 'test_label')])

filenames = glob.glob(osp.join(root, 'train_color/*.jpg'))
num_files = len(filenames)
file_index = np.random.choice(num_files, size=(int)(num_files * 0.2), replace=False)

print ("Moving files")
count = 0
for ind in file_index:
    file = filenames[ind]
    label_file = file[:-4] + '_instanceIds.png'
    label_file = label_file.replace('color', 'label')
    if count < num_files * 0.1:
        sp.call(['mv', file, osp.join(root, 'val_color')])
        sp.call(['mv', label_file, osp.join(root, 'val_label')])
    else:
        sp.call(['mv', file, osp.join(root, 'test_color')])
        sp.call(['mv', label_file, osp.join(root, 'test_label')])
    count += 1

print("Done")
