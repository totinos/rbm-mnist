#!/usr/bin/python

###########################################################################
#                                                                         #
#  FILENAME    -- mnist.py                                                #
#                                                                         #
#  AUTHOR      -- Sam Brown                                               #
#                                                                         #
#  DESCRIPTION -- This file implements functions that allow the user to   #
#                 parse and manipulate the MNIST handwritten digit        #
#                 dataset.                                                #
#                                                                         #
###########################################################################


from array import array
import struct
import numpy as np
import matplotlib.pyplot as plt



def load_images(num_digits=0, training=True):
    """Load MNIST digit images and labels into local memory.

    Args:
        num_digits - Describes the number of images to load (0 means all)
        training - <True> loads training set, <False> loads test set

    Returns:
        images - A numpy array of dim. [num_digits] x [pixels].
        labels - A numpy array of dim. [num_digits] x [1].

    The following files should be located in the relative folder images/
        - train-images-idx3-ubyte
        - train-labels-idx1-ubyte
        - t10k-images-idx3-ubyte
        - t10k-labels-idx1-ubyte
    """

    # Determine the set of binary data files to parse
    if training == True:
        ubytes = 'images/train-images-idx3-ubyte'
        labels = 'images/train-labels-idx1-ubyte'
    else:
        ubytes = 'images/t10k-images-idx3-ubyte'
        labels = 'images/t10k-labels-idx1-ubyte'

    # Unpack the binary image data and store it in a numpy array
    # Normalize the data to the interval [0,1)
    with open(ubytes, 'rb') as f:
        print('Unpacking binary image data...')
        magic_number, num_images, rows, cols = struct.unpack('>iiii', f.read(16))
        if (num_digits == 0 or num_digits > num_images):
            num_digits = num_images
        pixels = int(rows * cols)
        data = array('B', f.read(int(pixels * num_digits)))
        images = np.zeros((num_digits, pixels), dtype=np.uint8)
        for i in range(num_digits):
            start = int(i * pixels)
            end = int((i+1) * pixels)
            images[i] = np.array(data[start:end])
        images = np.true_divide(images, 255)
        print('Done!')

    # Unpack the binary label data and store it in a numpy array
    with open(labels, 'rb') as f:
        print('Unpacking binary label data...')
        magic_number, num_labels = struct.unpack('>ii', f.read(8))
        data = array('B', f.read(int(num_digits)))
        labels = np.array(data, dtype=np.uint8)
        print('Done!')

    return images, labels



def get_instances(digit, images, labels):
    """Get an array of images of the specified digit.

    Args:
        digit - A label describing the digit for which to look.
        images - A numpy array of dim. [num_digits] x [pixels].
        labels - A numpy array of dim. [num_digits] x [1].

    Returns:
        images - A numpy array of all digits matching the given label.
    """

    indices = []
    for i in range(len(labels)):
        if labels[i] == digit:
            indices.append(i)
    return images[indices]



def exclude_instances(digit, images, labels):
    """Get an array of images of all digits excepting the one specified.

    Args:
        digit - A label describing the digit to be excluded from the dataset.
        images - A numpy array of dim. [num_digits] x [pixels]
        labels - A numpy array of dim. [num_digits] x [1].
    """

    indices = []
    for i in range(len(labels)):
        if labels[i] != digit:
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

    twos = get_instances(2, images, labels)
    print('Shape of twos array: ', twos.shape)

    save_image(twos[0], 'fig.png')
