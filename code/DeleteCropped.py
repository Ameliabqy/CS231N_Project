from PIL import Image
import os
import os.path as osp
import subprocess as sp
import random
import numpy as np

if __name__ == '__main__':
    root = '../data'
    image = '../data/train_color'
    image_label = '../data/train_label'
    saved_location = image.replace('train_color', 'cropped_train_color/')
    saved_location_label = image.replace('train_color', 'cropped_train_label/')
    
    directory = os.fsencode(image)
    new_directory = os.fsencode(saved_location)
    index = 0
    for file in os.listdir(new_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"): 
            try:
                os.remove(os.path.join(image, filename[:]))
            except:
                pass
        else:
            continue
            
            
