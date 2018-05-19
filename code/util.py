import os
import tensorflow as tf
import numpy as np
import glob
import math
import random
import timeit
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from scipy import stats
import gc
import psutil
import shutil
from PIL import Image

import keras


def print_graph_size():
    i = 0
    for op in tf.get_default_graph().get_operations():
        i += 1
    print("Number of Graph Ops Stored: ", i)
    
    
def show_image(hp, image_tensor):
    with tf.Session() as session:
        image_model = tf.global_variables_initializer()
        session.run(hp.image_model)
        result = session.run(image_tensor)
    plt.imshow(result)
    plt.show()    
    
    
def check_accuracy(hp, session, outputs, labels):
    int_outputs = np.squeeze(outputs.astype(int))
    weight_outputs = int_outputs.copy()
    weight_labels = np.squeeze(labels.copy())

    outputs_list = []
    labels_list = []
    mean_list = []
    for i in range(weight_labels.shape[0]):
        
        label = weight_labels[i]
        mask_index = np.where(label != 0)
        label = label[mask_index]
        labels_list.append(label)
        
        output = weight_outputs[i]
        output = output[mask_index]
        outputs_list.append(output)

        mean_list.append(np.mean(np.equal(output, label).astype(int)))
        
    if hp.verbose:
        best_pair = np.argmax(mean_list, axis=0)
        overall_mean = np.mean(mean_list)

        print("WEIGHTED IMAGES min: %d | max: %d | mean: %d"
              %(np.amin(weight_outputs),np.amax(weight_outputs), np.mean(weight_outputs)))

        print("INT IMAGES min: %d | max: %d | mean: %d"
              %(np.amin(int_outputs),np.amax(int_outputs), np.mean(int_outputs)))

        print("WEIGHTED LABELS min: %d | max: %d | mean: %d"
              %(np.amin(weight_labels),np.amax(weight_labels), np.mean(weight_labels)))

        print("LABELS min: %d | max: %d | mean: %d"
            %(np.amin(labels),np.amax(labels), np.mean(labels)))

        print("best pair: ", best_pair)
        print("means list", mean_list)

    if hp.print_images:
        result = session.run(labels[best_pair]/tf.reduce_max(labels[best_pair]))
        plt.imshow(result, cmap='gray')
        plt.show()
    #     plt.figure
    #     result = session.run(tf.convert_to_tensor(labels[best_pair]))
    #     plt.imshow(result)
    #     plt.show()

        plt.figure
        result = session.run(outputs[best_pair]/tf.reduce_max(outputs[best_pair]))
        plt.imshow(result, cmap='gray')
        plt.show()
    #     plt.figure
    #     result = session.run(tf.convert_to_tensor(outputs[best_pair]))
    #     plt.imshow(result)
    #     plt.show()

    return overall_mean, np.mean(np.equal(int_outputs, labels).astype(int))



# ----------------------------------------- SAVE FUNCTIONS ----------------------------------------------- #

def create_saver(sess):
    """
    Creates setup for model saver.
    If a model already exists in SavedModel, it copies this over to SavedModel-Copy.
    Then, delete SavedModel so we can write to it.
    """
        
    if os.path.isdir('./SavedModel-COPY/'):
        shutil.rmtree('./SavedModel-COPY/')
    if os.path.isdir('./SavedModel/'):
        shutil.copytree('./SavedModel/', './SavedModel-COPY/')
        shutil.rmtree('./SavedModel/')
    builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
#     saver = tf.train.Saver() 
    return builder


def save_model(sess, builder):
    """
    Saves model to SavedModel.
    If SavedModel already exists, copies this to SavedModel-Copy. Then deletes SavedModel so we can write to it.
    """
    
    if os.path.isdir('./SavedModel-COPY/'):
        shutil.rmtree('./SavedModel-COPY/')
    if os.path.isdir('./SavedModel/'):
        shutil.copytree('./SavedModel/', './SavedModel-COPY/')
        shutil.rmtree('./SavedModel/')
    builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
    builder.add_meta_graph_and_variables(sess,
                           [tf.saved_model.tag_constants.TRAINING],
                           signature_def_map=None,
                           assets_collection=None)
    builder.save()
    
   
def load_saved_model(sess):
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './SavedModel/')



    
    
# ------------------------------------------ MEMORY FUNCTIONS --------------------------------------------- #


# This function from StackOverflow user "CodeGench"
# https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python

def print_memory():
    CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))

    #print results
    print("CPU Usage = " + CPU_Pct)
    mem=str(os.popen('free -t -m').readlines())
    """
    Get a whole line of memory output, it will be something like below
    ['             total       used       free     shared    buffers     cached\n', 
    'Mem:           925        591        334         14         30        355\n', 
    '-/+ buffers/cache:        205        719\n', 
    'Swap:           99          0         99\n', 
    'Total:        1025        591        434\n']
     So, we need total memory, usage and free memory.
     We should find the index of capital T which is unique at this string
    """
    T_ind=mem.index('T')
    """
    Than, we can recreate the string with this information. After T we have,
    "Total:        " which has 14 characters, so we can start from index of T +14
    and last 4 characters are also not necessary.
    We can create a new sub-string using this information
    """
    mem_G=mem[T_ind+14:-4]
    """
    The result will be like
    1025        603        422
    we need to find first index of the first space, and we can start our substring
    from from 0 to this index number, this will give us the string of total memory
    """
    S1_ind=mem_G.index(' ')
    mem_T=mem_G[0:S1_ind]
    """
    Similarly we will create a new sub-string, which will start at the second value. 
    The resulting string will be like
    603        422
    Again, we should find the index of first space and than the 
    take the Used Memory and Free memory.
    """
    mem_G1=mem_G[S1_ind+8:]
    S2_ind=mem_G1.index(' ')
    mem_U=mem_G1[0:S2_ind]

    mem_F=mem_G1[S2_ind+8:]
    print ('Summary = ', mem_G)
    print ('Total Memory = ', mem_T, ' MB')
    print ('Used Memory = ', mem_U, ' MB')
    print ('Free Memory = ', mem_F, ' MB')