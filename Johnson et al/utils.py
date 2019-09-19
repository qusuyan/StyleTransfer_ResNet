import numpy as np
import os, sys

import scipy.io
import scipy.misc

NOISE_RATIO = 0.6
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 400
COLOR_CHANNELS = 3

def add_noise(image, noise_ratio = NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + image * (1 - noise_ratio)
    
    return input_image

def open_image(path):
	image = scipy.misc.imread(path, mode='RGB')
	resized_image = scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
	return resized_image

def save_image(path, image):
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def normalize(image):
    image = image - MEANS
    return image

def denormalize(image):
    image = image + MEANS
    return image

def reshape(image):
	image = np.reshape(image, ((1,) + image.shape))
	return image

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]