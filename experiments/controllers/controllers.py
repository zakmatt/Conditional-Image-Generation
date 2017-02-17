#!/usr/bin/env python3
import cv2
import glob
import numpy as np
import os
import theano

EXTENSIONS = ['png', 'jpg']
theano.config.floatX = 'float32'

def load_data(directories, mask = None):
    
    def load_images(directory):
        image_list = []
        for filename in glob.glob(directory):
            ext = filename.split('.')
            if ext[-1].lower() not in EXTENSIONS:
                continue
            image = cv2.imread(filename)
            image_list.append(image)
            
        return image_list
    
    def shared_dataset(dataset, mask, borrow = True):
        """
        Keep dataset in shared variables. This trick allows Theano
        to copy data into the GPU memory, when the code is run on the GPU.
        Since copying everything to the GPU is an overwhelming task, we'll
        copy mini-batches.
        """

        if mask == None:
            scale = 0.25
            image = dataset[0]
            mask = np.ones(image.shape)
            l = int(scale * image.shape[0]) - 1
            u = int((1.0 - scale) * image.shape[0])
            print(l,u)
            mask[l:u, l:u, :] = 0.0

        # create input and output of the nn
        dataset_x = dataset * mask
        dataset_y = dataset * (1 - mask)

        # make shared variables of input and output
        shared_x = theano.shared(np.asanyarray(dataset_x,
                                               dtype = theano.config.floatX))
        shared_y = theano.shared(np.asanyarray(dataset_y,
                                               dtype = theano.config.floatX))
        
        return shared_x, shared_y
    
    image_list = []
    for directory in directories:
        images = load_images(directory)
        if len(image_list) == 0:
            image_list = np.array(images.copy())
        else:
            image_list = np.concatenate((image_list, images.copy()), axis = 0)

    dataset_x, dataset_y = shared_dataset(image_list, mask)
    return dataset_x, dataset_y
        
def search_dir(directory):
    directories = []
    for folder_name in os.listdir(directory):
        dir = os.path.join(directory, folder_name)
        if os.path.isfile(dir):
            continue
        dir = os.path.join(dir, '*')
        directories.append(dir)
    return directories
        
        
if __name__ == '__main__':
    path = '../dataset/'
    directories = search_dir(path)
    dataset_x, dataset_y = load_data(directories)