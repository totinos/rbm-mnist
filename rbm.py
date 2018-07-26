import numpy as np
import os
import sys
import mnist

from time import time

def rbm(dataset, num_hidden, learning_rate, max_epochs, batch_size):
    """Train a RBM where the input is binary or real over interval [0,1].

    The training algorithm splits the dataset into minibatches and uses
    momentum when updating the weight values, as Hinton does in his MATLAB
    implementation.

    Args:
        dataset - A numpy array of dim. [num_examples] x [num_inputs].
        num_hidden - The number of hidden neurons in the RBM implementation.
        learning_rate - Used when updating weights (ref. Geoff Hinton).
        max_epochs - The number of epochs to train the RBM network.
        batch_size - The number of training examples per batch.

    Returns:
        weights - A numpy array of dim. [num_visible] x [num_hidden].
        v_bias - A numpy array of dim. [num_visible] x [1].
        h_bias - A numpy array of dim. [num_hidden] x [1].
    """

    num_examples = dataset.shape[0]
    num_visible = dataset.shape[1]

    start_time = time()

    num_batches = num_examples // batch_size

    # Initialize the weight matrices
    weights = 0.1 * np.random.randn(num_visible, num_hidden)
    v_bias = np.zeros((1, num_visible))
    h_bias = -4.0 * np.ones((1, num_hidden))
    #h_bias = np.zeros((1, num_hidden))

    # Initialize the weight incremental update matrices
    w_inc = np.zeros((num_visible, num_hidden))
    v_inc = np.zeros((1, num_visible))
    h_inc = np.zeros((1, num_hidden))

    for epoch in range(max_epochs):
        error = 0
        for batch in range(num_batches):

            ####### POSITIVE CONTRASTIVE DIVERGENCE PHASE ######
            
            # Get next batch of data
            start = int(batch * batch_size)
            end = int((batch+1) * batch_size)
            data = dataset[start:end]
            
            # In this matrix, m[i,j] is prob h[j] = 1 given example v[i]
            # Dim. [num_examples] x [num_hidden]
            pos_hid_probs = logistic(data, weights, h_bias)

            # Sample the states of the hidden units based on pos_hid_probs
            pos_hid_states = pos_hid_probs > np.random.rand(batch_size, num_hidden)

            # Positive phase products
            pos_prods = np.dot(data.T, pos_hid_probs)

            # Activation values needed to update biases
            pos_hid_act = np.sum(pos_hid_probs, axis=0)
            pos_vis_act = np.sum(data, axis=0)

            ###### NEGATIVE CONTRASTIVE DIVERGENCE PHASE ######

            # Reconstruct the data by sampling the vsible states from hidden states
            neg_data = logistic(pos_hid_states, weights.T, v_bias)

            # Sample hidden states from visible states
            neg_hid_probs = logistic(neg_data, weights, h_bias)

            # Negative phase products
            neg_prods = np.dot(neg_data.T, neg_hid_probs)

            # Activation values needed to update biases
            neg_hid_act = np.sum(neg_hid_probs, axis=0)
            neg_vis_act = np.sum(neg_data, axis=0)

            ###### UPDATE WEIGHT VALUES ######

            # Set momentum as per Hinton's methods
            m = 0.5 if epoch > 5 else 0.9

            # Update the weights
            x = learning_rate/batch_size
            w_inc = (w_inc * m) + x * (pos_prods - neg_prods)
            v_inc = (v_inc * m) + x * (pos_vis_act - neg_vis_act)
            h_inc = (h_inc * m) + x * (pos_hid_act - neg_hid_act)

            weights += w_inc
            v_bias += v_inc
            h_bias += h_inc

            error += np.sum((data - neg_data) ** 2)

        time_elapsed = time() - start_time
        print('Epoch %4d completed. Reconstruction error is %0.2f. Sim time (s): %0.2f' % (epoch+1, error, time_elapsed))

    print('Training completed.')
    return weights, v_bias, h_bias


def logistic(x, w, b):
    """A sigmoid relationship for RBM activation function.

    Args:
        x - A numpy array of states.
        w - A numpy array of weights.
        b - A numpy array of biases.

    Returns:
        A numpy array after the logistic function is applied.
    """

    xw = np.dot(x, w)
    # replicated_b = np.tile(b, (x.shape[0], 1))
    return 1.0 / (1 + np.exp(- xw - b))

def lookup(x, w, b):
    """A stochastic neuron activation function model.

    Args:
        x - A numpy array of states.
        w - A numpy array of weights.
        b - A numpy array of biases.
    """
    print('LOOKUP HERE.')

if __name__ == '__main__':
    if len(sys.argv) == 6:
        num_examples = int(sys.argv[1])
        num_hidden = int(sys.argv[2])
        max_epochs = int(sys.argv[3])
        learning_rate = float(sys.argv[4])
        batch_size = int(sys.argv[5])

        images, labels = mnist.load_images(num_examples, True)
        w, a, b = rbm(images, num_hidden, learning_rate, max_epochs, batch_size)
    elif len(sys.argv) == 1:
        print('Training RBM on 1000 examples over 50 epochs with learning rate of 0.1 and 100 hidden neurons.')
        images, labels = mnist.load_images(1000, True)
        w, a, b = rbm(images, 100, 0.1, 50, batch_size=100)
    else:
        print('Usage: python rbm.py [num_examples] [num_hidden] [max_epochs] [learning_rate] [batch_size]')
        exit(1)
