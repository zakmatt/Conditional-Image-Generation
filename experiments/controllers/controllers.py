#!/usr/bin/env python3
import argparse
import imageio
import glob
import numpy as np
from os import listdir
from os.path import isfile, join
import theano
from tqdm import tqdm

def load_data(directory):
    # rescale to 0 - 1, since generator output is tanh
    dataset = np.load(directory).items()[0][1] / 255.
    dataset = np.array(dataset, dtype = np.float32)
    # swap axes to (number of channels, height, width)
    # primarely its (height, width, number of channels)
    dataset = np.swapaxes(dataset, 1, 3)
    dataset = np.swapaxes(dataset, 2, 3)
    
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
    directories = [join(directory, file) for file in listdir(directory) if
                   isfile(join(directory, file)) and file[-3:] == 'npz']
    return directories

def prepare_data(mode, home):
    path = 'inpainting/val2014/' if mode == 'validate' else 'inpainting/train2014/'
    save_path = 'images.%s' % ('validate' if mode == 'validate' else 'train')
    images = []
    index = 0
    for fname in tqdm(glob.glob(join(home, path) + '*')):
        img = imageio.imread(fname)
        if img.shape == (64, 64, 3) and img.dtype == np.uint8:
            if np.array_equal(img[:, :, 0], img[:, :, 1]) and \
                             np.array_equal(img[:, :, 0], img[:, :, 2]):
                continue
            images.append(img)
            if len(images) >= 10000:
                images = np.array(images)
                np.savez_compressed('%s_%d.npz' % (join(home, save_path), index), images)
                images = []
                index += 1
    if len(images) > 0:
        images = np.array(images)
        np.savez_compressed('%s_%d.npz' % (join(home, save_path), index), images)
        
        
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