#!/usr/bin/env python

"""

ADAPTED FROM BELOW BY RYAN SCHORK. PLEASE VISIT THE LASAGNE LINKS BELOW BEFORE
USE.

Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import random
import theano
import theano.tensor as T

import lasagne
from sklearn import metrics

class MLP(object):

    def __init__(self,
                 pickle_dataset = 'numerai.pkl.gz',
                 hidden_layers_sizes = [25,15,10,5],
                 nonlin = lasagne.nonlinearities.rectify,
                 out_layer = lasagne.nonlinearities.softmax,
                 drop_input = .1,
                 drop_hidden = .1,
                 learning_rate = .05,
                 batchsize = 100,
                 num_epochs = 500,
                 momentum = 0.9,
                 early_stopping_threshold = 25,
                 rand_seed = 8675309,
                 save = False,
                 verbose = True,
                 enforce_train_supremacy = False,
                 metric = 'class'):

        # Create neural network model (depending on first command line parameter)

        random.seed(rand_seed)
        np.random.seed(rand_seed)

        print("\r\n\r\nBuilding model and compiling functions with random seed: {}...".format(rand_seed))

        self.pickle_dataset = pickle_dataset
        self.hidden_layers_sizes = hidden_layers_sizes
        self.nonlin = nonlin
        self.out_layer = out_layer
        self.drop_input = float(drop_input)
        self.drop_hidden = float(drop_hidden)
        self.learning_rate = float(learning_rate)
        self.batchsize = int(batchsize)
        self.num_epochs = int(num_epochs)
        self.momentum = float(momentum)
        self.early_stopping_threshold = int(early_stopping_threshold)
        self.rand_seed = rand_seed
        self.enforce_train_supremacy = enforce_train_supremacy
        self.save = save
        self.verbose = verbose
        if metric not in ['class', 'auc']:
            print('pick class or auc for metric you doofus')
            exit()
        self.metric = metric

    def returnAUC(self, predictions, target):
        fpr, tpr, thresholds = metrics.roc_curve(target, predictions, pos_label=max(target))
        auc = metrics.auc(fpr, tpr)
        return auc


    def load_dataset(self, dataset = None):

        import gzip
        import cPickle

        if dataset == None:
            dataset = self.pickle_dataset

        ''' Loads the dataset

        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        '''

        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set, score_set = cPickle.load(f)
        f.close()

        return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1], score_set
        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.


    # ##################### Build the neural network model #######################
    # This script supports three types of models. For each one, we define a
    # function that takes a Theano variable representing the input and returns
    # the output layer of a neural network model built in Lasagne.

    def build_custom_mlp(self, input_var=None):

        depth = len(self.hidden_layers_sizes)

        # By default, this creates the same network as `build_mlp`, but it can be
        # customized with respect to the number and size of hidden layers. This
        # mostly showcases how creating a network in Python code can be a lot more
        # flexible than a configuration file. Note that to make the code easier,
        # all the layers are just called `network` -- there is no need to give them
        # different names if all we return is the last one we created anyway; we
        # just used different names above for clarity.

        # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
        network = lasagne.layers.InputLayer(shape=(None, self.n_ins),
                                            input_var=input_var)
        if self.drop_input:
            network = lasagne.layers.dropout(network, p=self.drop_input)
        # Hidden layers and dropout:
        nonlin = self.nonlin
        for index in range(0, depth):
            network = lasagne.layers.DenseLayer(
                    network, self.hidden_layers_sizes[index], nonlinearity=self.nonlin)
            if self.drop_hidden:
                network = lasagne.layers.dropout(network, p=self.drop_hidden)
        # Output layer:
        network = lasagne.layers.DenseLayer(network, self.n_outs, nonlinearity=self.out_layer)
        return network

    # ############################# Batch iterator ###############################
    # This is just a simple helper function iterating over training data in
    # mini-batches of a particular size, optionally in random order. It assumes
    # data is available as numpy arrays. For big datasets, you could load numpy
    # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    # own custom data iteration function. For small datasets, you can also copy
    # them to GPU at once for slightly improved performance. This would involve
    # several changes in the main program, though, and is not demonstrated here.

    def iterate_minibatches(self, inputs, targets, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield inputs[excerpt], targets[excerpt]


    # ############################## Main program ################################
    # Everything else will be handled in our main program now. We could pull out
    # more functions to better separate the code, but it wouldn't make it any
    # easier to read.

    def save_predictions_raw(self):
        print('...saving ' + 'MLP_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '_' + 'pred_raw.csv')
        np.savetxt('MLP_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '_' + 'pred_raw.csv', self.save_predictions, delimiter=",")

    def train(self):
        # Load the dataset
        print("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test, X_score = self.load_dataset()

        self.n_ins = int(X_train.shape[1])
        self.n_outs = int(max(y_train) + 1)

        # Prepare Theano variables for inputs and targets
        #input_var = T.tensor4('inputs')
        input_var = T.matrix('inputs')
        target_var = T.ivector('targets')

        network = self.build_custom_mlp(input_var)
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        train_prediction = lasagne.layers.get_output(network, deterministic=True)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=self.learning_rate, momentum=self.momentum)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

        train_fn_acc = theano.function([input_var, target_var], train_acc, allow_input_downcast=True)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

        scoring_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:

        maximum_valid_accuracy = 0.0
        ass_training_accuracy = 0.0
        ass_testing_accuracy = 0.0
        max_epoch = 0
        epoch_number = 0

        early_stop_count = 0

        for epoch in range(self.num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_acc_measure = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_acc_measure += train_fn_acc(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch in self.iterate_minibatches(X_val, y_val, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # During training, we compute and print the test error (but we don't peek!):
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in self.iterate_minibatches(X_test, y_test, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1

            early_stop_count += 1

            # if (val_acc / val_batches > maximum_valid_accuracy):
            if self.enforce_train_supremacy:
                if ((train_acc_measure / train_batches) > (val_acc / val_batches)):
                    train_gate = True
                else:
                    train_gate = False
            else:
                train_gate = True

            if (val_acc / val_batches > maximum_valid_accuracy) and train_gate:
                maximum_valid_accuracy = val_acc / val_batches
                ass_training_accuracy = train_acc_measure / train_batches
                ass_testing_accuracy = test_acc / test_batches
                max_epoch = epoch + 1

                self.save_params = lasagne.layers.get_all_param_values(network)
                self.save_predictions = scoring_fn(X_score)

                early_stop_count = 0

            if self.verbose == True:

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, self.num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  training accuracy:\t\t{:.2f} %".format(
                    train_acc_measure / train_batches * 100))
                print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))
                print("  testing accuracy:\t\t{:.2f} %".format(
                    test_acc / test_batches * 100))
                if epoch > 10 and epoch%10==0:
                    print("best validation accuracy at epoch {}: {:.2f} % (training accuracy: {:.2f} %)".format(max_epoch,
                        maximum_valid_accuracy * 100, ass_training_accuracy))

            epoch_number += 1

            if self.early_stopping_threshold and early_stop_count > self.early_stopping_threshold:
                print("\r\n\r\n NOTE: EARLY STOPPING at EPOCH {} OUT OF {} DUE TO LACK OF VALIDATION IMPROVEMENT...\r\n\r\n".format(epoch_number, self.num_epochs))
                break

        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
        print("  best validation accuracy at epoch {} (out of {} total epoch): {:.2f} % (training accuracy: {:.2f} %, testing accuracy: {:.2f} %)".format(max_epoch, epoch_number,
            maximum_valid_accuracy * 100, ass_training_accuracy * 100, ass_testing_accuracy * 100))

        if self.save == True:
            output = 'MLP_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '.npz'
            np.savez(output, self.save_params)

        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':

    mlp_fun = MLP(rand_seed=8675444, save=False, verbose=True, early_stopping_threshold = False, enforce_train_supremacy=True, num_epochs=500)
    mlp_fun.train()
    mlp_fun.save_predictions_raw()
    # ensemble = mlp_fun.save_predictions
    #
    # count = 1
    #
    # for rand_seed_test in range(0,100):
    #     count += 1
    #     mlp_fun = MLP(rand_seed=rand_seed_test, save=False, verbose=False, early_stopping_threshold = 100)
    #     mlp_fun.train()
    #     ensemble += mlp_fun.save_predictions
    #
    # print(count)
    #
    # ensemble = ensemble / count
    #
    # np.savetxt('titanic_ensemble1.csv',ensemble, delimiter=",")
