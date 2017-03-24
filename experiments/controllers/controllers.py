#!/usr/bin/env python3
import argparse
import imageio
import glob
import numpy as np
import os
import theano
from tqdm import tqdm

theano.config.floatX = 'float32'

def load_data(directory):
    
    def shared_dataset(dataset, borrow = True):
        """
        Keep dataset in shared variables. This trick allows Theano
        to copy data into the GPU memory, when the code is run on the GPU.
        Since copying everything to the GPU is an overwhelming task, we'll
        copy mini-batches.
        """
        # make shared variables of input and output
        shared_dataset = theano.shared(np.asanyarray(dataset,
                                               dtype = theano.config.floatX),
                                        borrow = borrow)
        
        return shared_dataset
    
    # rescale to 0 - 1, since generator output is tanh
    dataset = np.load(directory).items()[0][1] / 255.
    # swap axes to (number of channels, height, width)
    # primarely its (height, width, number of channels)
    dataset = np.swapaxes(dataset, 1, 3)
    dataset = np.swapaxes(dataset, 2, 3)
    dataset = shared_dataset(dataset)
    
    return dataset

def save_data(dataset, path):
    dataset = rescale(dataset)
    dataset = np.swapaxes(dataset, 2, 3)
    dataset = np.swapaxes(dataset, 1, 3)
    np.save(path, dataset)
    
def fill_missing_part(full_images, generated):
    full_images[:, :, 16:48, 16:48] = generated[:, :, 16:48, 16:48]
    return full_images

def rescale(image):
    image = image.astype('float32')
    current_min = np.min(image)
    current_max = np.max(image)
    image = (image - current_min)/(current_max - current_min) * 255
    return image.astype('uint8')

def search_dir(directory):
    directories = [os.path.join(directory, '*')]
    for folder_name in os.listdir(directory):
        dir = os.path.join(directory, folder_name)
        if os.path.isfile(dir):
            continue
        dir = os.path.join(dir, '*')
        directories.append(dir)
    return directories

def prepare_data(mode, home):
    path = 'inpainting/val2014/' if mode == 'validate' else 'inpainting/train2014/'
    save_path = 'images.%s' % ('validate.npz' if mode == 'validate' else 'train.npz')
    images = []
    for fname in tqdm(glob.glob(os.path.join(home, path) + '*')):
        img = imageio.imread(fname)
        if img.shape == (64, 64, 3) and img.dtype == np.uint8:
            if np.array_equal(img[:, :, 0], img[:, :, 1]) and \
                             np.array_equal(img[:, :, 0], img[:, :, 2]):
                continue
            images.append(img)
    images = np.array(images)
    np.savez_compressed('%s%s' % os.path.join(home, save_path), images)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_directory",
                        help = 'Directory to the data',
                        required = True)
    args = parser.parse_args()
    directory = args.data_directory
    prepare_data('train', directory)
    prepare_data('validate', directory)