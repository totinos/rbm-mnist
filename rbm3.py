#!/usr/bin/python

###########################################################################
#                                                                         #
#  FILENAME    -- rbm3.py                                                 #
#                                                                         #
#  AUTHOR      -- Sam Brown                                               #
#                                                                         #
#  DESCRIPTION -- This file implements functions that allow the user to   #
#                 create and train a Restricted Boltzmann Machine using   #
#                 any dataset.                                            #
#                                                                         #
###########################################################################


import numpy as np
import os
import sys
import mnist
from time import time
import matplotlib.pyplot as plt



class RBM:



    def __init__(self, num_visible, num_hidden, act_func='logistic'):
        """Initializes an RBM with the given dimensions.

        Args:
            dataset - A numpy array of dim. [num_examples] x [num_inputs].
            num_hidden - The number of hidden neurons in the RBM implementation.
            act_func - The type of activation function the RBM should use.
        """

        print('Initializing network...  ', end='')
        sys.stdout.flush()

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        #self.reconstructed = np.zeros((self.num_examples, self.num_visible))

        self.weights = 0.1 * np.random.randn(num_visible, num_hidden)
        self.v_bias = np.zeros((1, num_visible))
        self.h_bias = -4.0 * np.ones((1, num_hidden))

        self.w_inc = np.zeros((num_visible, num_hidden))
        self.v_inc = np.zeros((1, num_visible))
        self.h_inc = np.zeros((1, num_hidden))

        self.act_func = self.lookup if act_func == 'lookup' else self.logistic

        print('Done!')
        return



    def train(self, dataset, learning_rate=0.1, max_epochs=50, batch_size=100):
        """Train a RBM where the input is binary or real over interval [0,1].

        The training algorithm splits the dataset into minibatches and uses
        momentum when updating the weight values, as Hinton does in his MATLAB
        implementation.

        Args:
            learning_rate - Used when updating weights (ref. Geoff Hinton).
            max_epochs - The number of epochs to train the RBM network.
            batch_size - The number of training examples per batch.
        """

        num_examples = dataset.shape[0]
        num_batches = num_examples // batch_size

        self.reconstructed = None
        self.reconstructed = np.zeros((num_batches, self.num_visible))

        # Train the RBM over given # of epochs using Contrastive Divergence
        start_time = time()
        for epoch in range(max_epochs):
            error = 0
            momentum = 0
            for batch in range(num_batches):
                start = int(batch * batch_size)
                end = int((batch+1) * batch_size)
                data = dataset[start:end]

                ############# POSITIVE #############
                pos_hid_probs = self.act_func(data, self.weights, self.h_bias)
                rand_states = np.random.rand(batch_size, self.num_hidden)
                pos_hid_states = pos_hid_probs > rand_states
                pos_prods = np.dot(data.T, pos_hid_probs)
                pos_hid_act = np.sum(pos_hid_probs, axis=0)
                pos_vis_act = np.sum(data, axis=0)

                ############# NEGATIVE #############
                neg_data = self.act_func(pos_hid_states, self.weights.T, self.v_bias)
                neg_hid_probs = self.act_func(neg_data, self.weights, self.h_bias)
                neg_prods = np.dot(neg_data.T, neg_hid_probs)
                neg_hid_act = np.sum(neg_hid_probs, axis=0)
                neg_vis_act = np.sum(neg_data, axis=0)

                #############  UPDATE  #############
                m = 0.5 if epoch > 5 else 0.9
                x = learning_rate/batch_size
                self.w_inc = (self.w_inc * m) + x*(pos_prods - neg_prods)
                self.v_inc = (self.v_inc * m) + x*(pos_vis_act - neg_vis_act)
                self.h_inc = (self.h_inc * m) + x*(pos_hid_act - neg_hid_act)
                self.weights += self.w_inc
                self.v_bias += self.v_inc
                self.h_bias += self.h_inc

                error += np.sum((data - neg_data) ** 2)

                ############ RECONSTRUCT ###########
                self.reconstructed[batch] = neg_data[0]

            # TODO ---> ADD programmable sampling frequency
            # TODO ---> ADD a nice progress bar for convenience??
            #time_elapsed = time() - start_time
            if max_epochs < 10 or epoch % (max_epochs // 10) == 0:
                print('Epoch %4d -> Reconstruction error: %0.2f.' % (epoch, error))
                if not os.path.exists('reconstructed'):
                    os.makedirs('reconstructed')
                filename = 'reconstructed/{:04}.png'.format(epoch)
                self.save_reconstructed_images(filename, 28, 28)

        time_elapsed = time() - start_time
        print('Training completed.')
        print('Time elapsed (s): %0.2f' % time_elapsed)
        return



    def logistic(self, data, weights, biases):
        """Logistic activation function"""
        state_weight_prods = np.dot(data, weights)
        # replicated_bias = np.tile(biases, (data.shape[0], 1))
        return 1.0 / (1 + np.exp(- state_weight_prods - biases))

    def lookup(self, data, weights, biases):
        """Stochastic neuron activation function lookup table."""
        return




    def save_reconstructed_images(self, filename, rows, cols):
        """Saves an image representing the reconstructed data to disk.

        Args:
            filename - A location to save the image file.
        """
        # print(filename)
        num_images = self.reconstructed.shape[0]
        images = np.zeros((rows, cols*num_images))
        for i in range(num_images):
            start = int(i * cols)
            end = int((i+1) * cols)
            images[0:rows, start:end] = self.reconstructed[i].reshape((rows, cols))

        plt.imshow(images, cmap='gray', aspect='equal', interpolation='none')
        #plt.show()
        plt.savefig(filename)
        return



    def reconstruct_image(self, input_vector):
        """Reconstructs the network's impression of the input image.

        The hidden states are sampled after the inputs are introduced, and
        then the visible states are sampled based on the propagation of the
        hidden states backward through the network.

        Args:
            input_vector - A numpy array of dimension [num_inputs].
        """

        pos_hid_probs = self.act_func(input_vector, self.weights, self.h_bias)
        pos_hid_states = pos_hid_probs > np.random.rand(1, self.num_hidden)
        neg_data = self.act_func(pos_hid_states, self.weights.T, self.v_bias)
        return neg_data



    def save_weights(self, outfile='', title=''):
        """Save image with pictoral representation of weight distributions.

        Args:
            outfile - The filename for the image to be saved.
            title - The title for the image to be saved.
        """

        row = self.num_visible
        col = self.num_hidden
        # plt.imshow(self.weights, cmap='gray', aspect='auto', interpolation='none')
        # plt.show()

        # plt.imshow(self.reconstruct_image(self.dataset[0]).reshape((28,28)), cmap='gray', aspect='equal', interpolation='none')
        # plt.show()

        # plt.imshow(self.dataset[0].reshape((28,28)), cmap='gray', aspect='equal', interpolation='none')
        # plt.show()
        return
    


if __name__ == '__main__':
    if len(sys.argv) == 7:
        num_examples = int(sys.argv[1])
        num_hidden = int(sys.argv[2])
        max_epochs = int(sys.argv[3])
        learning_rate = float(sys.argv[4])
        batch_size = int(sys.argv[5])
        act_func = str(sys.argv[6])

        images, labels = mnist.load_images(num_examples, True)


        num_visible = images.shape[1]
        #num_examples = images.shape[0]


        rbm = RBM(num_visible, num_hidden, act_func)
        rbm.train(images, learning_rate, max_epochs, batch_size)
        rbm.save_weights("", "")
        rbm.save_reconstructed_images("", 28, 28)

    elif len(sys.argv) == 1:
        print('Training RBM on 1000 examples over 50 epochs with learning rate of 0.1 and 100 hidden neurons.')
        images, labels = mnist.load_images(1000, True)
        num_visible = images.shape[1]
        rbm = RBM(num_visible, 100, 'logistic')
        rbm.train(images, 0.1, 50, 100)

    else:
        print('Usage: python rbm.py [num_examples] [num_hidden] [max_epochs] [learning_rate] [batch_size] [act_func]')
        exit(1)
