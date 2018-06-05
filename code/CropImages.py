from PIL import Image
import os
import os.path as osp
import subprocess as sp
import random
import numpy as np

# Change this for how many pictures to generate 
NumCrop = 1000

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()
    
    
# divides labels by 1000 to get rid of instance numbers
def convert_labels(label_path):
    label_obj = Image.open(label_path)
    label2np = np.asarray(label_obj, dtype="int32")
    new_label = (label2np // 1000).astype("int8")
    image = Image.fromarray(new_label, mode='P')
    image.save(label_path)
 
 
if __name__ == '__main__':
    root = '../data'
    image = '../data/train_color'
    image_label = '../data/train_label'
    saved_location = image.replace('train_color', 'cropped_train_color/')
    saved_location_label = image.replace('train_color', 'cropped_train_label/')
    
    print ("Creating folders")
    sp.call(['mkdir', osp.join(root, 'cropped_train_color')])
    sp.call(['mkdir', osp.join(root, 'cropped_train_label')])
    
    directory = os.fsencode(image)
    new_directory = os.fsencode(saved_location)
    index = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"): 
            label_filename = filename[:-4] + '_instanceIds.png'
            label = Image.open(os.path.join(image_label, label_filename))
            label2np = np.asarray(label, dtype="int32" )
            if np.unique(label2np).shape[0] > 1:
                choice = random.choice(np.delete(np.unique(label2np), 0))
#                 print(choice)
                indices = np.where(label2np == choice)
                y = indices[0][0]
                x = indices[1][0]
            crop_coord = (x-192, y-192, x+192, y+192)
            crop(os.path.join(image, filename), crop_coord, os.path.join(saved_location, filename))
            crop(os.path.join(image_label, label_filename), crop_coord, os.path.join(saved_location_label, label_filename))
            convert_labels(os.path.join(saved_location_label, label_filename))
            index += 1
#             if (index == NumCrop):
#                 break
        else:
            continue
            
            
