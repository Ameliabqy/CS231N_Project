# Split_data.py
# written by Alice Li
# 5/15/18
# 
# This splits our training data into 85% train, 15% val

import numpy as np 
from numpy import random
np.random.seed(5)

import os
import os.path as osp
import glob
import subprocess as sp

root = '../data/cvpr-2018-autonomous-driving/'
ret = sp.call(['test', '-d', osp.join(root, 'val_color')])

if not ret:
	print ('Already split data, exiting:')
	quit()

print ("Creating folders")
sp.call(['mkdir', osp.join(root, 'val_color')])
sp.call(['mkdir', osp.join(root, 'val_label')])

filenames = glob.glob(osp.join(root, 'train_color/*.jpg'))
num_files = len(filenames)
file_index = np.random.choice(num_files, size=(int)(num_files * 0.15), replace=False)

print ("Moving files")
for ind in file_index:
	file = filenames[ind]
	label_file = file[:-4] + '_instanceIds.png'
	label_file = label_file.replace('color', 'label')
	sp.call(['mv', file, osp.join(root, 'val_color')])
	sp.call(['mv', label_file, osp.join(root, 'val_label')])

print("Done")
