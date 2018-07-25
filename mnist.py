#from pylab import *
from array import array
import struct
import numpy as np
import os

import matplotlib.pyplot as plt

def load_images(num_digits=0, training=True):
    """Load MNIST digit images and labels into local memory.
    Args:
        num_digits - Describes the number of images to load (0 means all)
        training - <True> loads training set, <False> loads test set

    The following files should be located in the relative folder images/
        - train-images-idx3-ubyte
        - train-labels-idx1-ubyte
        - t10k-images-idx3-ubyte
        - t10k-labels-idx1-ubyte
    """

    if training == True:
        ubytes = 'images/train-images-idx3-ubyte'
        labels = 'images/train-labels-idx1-ubyte'
    else:
        ubytes = 'images/t10k-images-idx3-ubyte'
        labels = 'images/t10k-labels-idx1-ubyte'

    with open(ubytes, 'rb') as f:
        magic_number, num_images, rows, cols = struct.unpack('>iiii', f.read(16))

        if (num_digits == 0 or num_digits > num_images):
            num_digits = num_images

        pixels = int(rows * cols)
        
        print(num_digits)
        print(pixels)

        data = array('B', f.read(int(pixels * num_digits)))
        images = np.zeros((num_digits, pixels), dtype=np.uint8)

        for i in range(num_digits):
            start = int(i * pixels)
            end = int((i+1) * pixels)
            images[i] = np.array(data[start:end])

        images = np.true_divide(images, 255)

    with open(labels, 'rb') as f:
        magic_number, num_labels = struct.unpack('>ii', f.read(8))
        data = array('B', f.read(int(num_digits)))
        labels = np.array(data, dtype=np.uint8)

    return images, labels

def filter_dataset(digit, images, labels):
    """Get an array of images of the specified digit.

    Args:
        digit - A label describing the digit for which to look.

    Returns:
        images - An array of all digits matching the given label.
    """

    indices = []
    for i in range(len(labels)):
        if labels[i] == digit:
            indices.append(i)
    return images[indices]

def save_image(digit, filename):
    """Display a single image given by an array of pixel values

    Args:
        digit - An array of pixel values normalized to 0-1 scale.
    """

    row = 28
    col = 28
    plt.imshow(digit.reshape((row, col)), cmap='gray', aspect='equal', interpolation='none')
    plt.axis('off')
    plt.savefig(filename)
    return

if __name__ == '__main__':
    images, labels = load_images(10, True)
    print('Shape of images array: ', images.shape)

    twos = filter_dataset(2, images, labels)
    print('Shape of twos array: ', twos.shape)

    save_image(twos[0], 'fig.png')
