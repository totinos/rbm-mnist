import numpy as np
import os
import sys


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

    # start_time = time()

    num_batches = num_examples // batch_size

    # Initialize the weight matrices
    weights = np.random.randn(num_visible, num_hidden)
    v_bias = np.zeros((1, num_visible)
    h_bias = -4.0 * np.ones((1, num_hidden))

    # Initialize the weight incremental update matrices
    w_inc = np.zeros((num_visible, num_hidden))
    v_inc = np.zeros((1, num_visible))
    h_inc = np.zeros((1, num_hidden))

    for epoch in range(epochs):
        error = 0
        for batch in range(num_batches):

            ####### POSITIVE CONTRASTIVE DIVERGENCE PHASE ######

            
