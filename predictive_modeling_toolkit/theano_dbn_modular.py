"""
"""
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM


# start-snippet-1
class DBN(object):
    """Deep Belief Network

    ADAPTED FROM BELOW BY RYAN SCHORK

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self,
                 numpy_rng=None,
                 theano_rng=None,
                 hidden_layers_sizes=[10],
                 finetune_lr=1,
                 pretraining_epochs=30,
                 pretrain_lr=.1,
                 k=1,
                 training_epochs=500,
                 pickle_dataset='numerai.pkl.gz',
                 batch_size=100,
                 L1 = 0.0000,
                 L2 = 0.0000,
                 activation = T.nnet.sigmoid,
                 patience = 100,
                 patience_increase = 20,
                 rand_seed = 8675309,
                 verbose = False,
                 enforce_train_supremacy = True):

        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.rand_seed = rand_seed
        if not numpy_rng:
            self.numpy_rng = numpy.random.RandomState(self.rand_seed)
        if not theano_rng:
            self.theano_rng = MRG_RandomStreams(self.rand_seed)
        self.hidden_layers_sizes = hidden_layers_sizes
        self.finetune_lr = finetune_lr
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.k = k
        self.training_epochs = training_epochs
        self.pickle_dataset = pickle_dataset
        self.batch_size = batch_size
        self.L1 = L1
        self.L2 = L2
        self.activation = activation
        self.patience = patience
        self.patience_increase = patience_increase
        self.verbose = verbose
        self.enforce_train_supremacy = enforce_train_supremacy

        self.train_set_x, self.train_set_y, self.valid_set_x, self.valid_set_y, self.test_set_x, self.test_set_y, self.score_set_x = self.load_dataset(self.pickle_dataset)

        assert self.n_layers > 0

        self.n_outs = int(max(self.train_set_y.eval()) + 1)
        self.n_ins = int(self.train_set_x.get_value(borrow=True).shape[1])

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = self.n_ins
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        activation=self.activation)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=self.numpy_rng,
                            theano_rng=self.theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=self.hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=self.n_outs)

        self.params.extend(self.logLayer.params)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small

        L1_cost = 0
        L2_cost = 0

        for layer in self.sigmoid_layers:
            L1_cost += abs(layer.W).sum()
            L2_cost += abs(layer.W ** 2).sum()

        L1_cost += abs(self.logLayer.W).sum()
        L2_cost += abs(self.logLayer.W ** 2).sum()

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) + (self.L1 * L1_cost) + (self.L2 * L2_cost)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)



    def load_dataset(self,dataset):
        # Load the dataset
        import gzip
        import cPickle
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set, score_set = cPickle.load(f)
        f.close()
        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.

        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        score_set_x = theano.shared(numpy.asarray(score_set, dtype=theano.config.floatX), borrow=True)

        return [train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, score_set_x]

    def pretraining_functions(self):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = self.train_set_x.get_value(borrow=True).shape[0] / self.batch_size
        # begining of a batch, given `index`
        batch_begin = index * self.batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + self.batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=self.k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: self.train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        # compute number of minibatches for training, validation and testing
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= self.batch_size
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= self.batch_size
        num_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        num_train_batches /= self.batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * self.finetune_lr))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: self.train_set_x[
                    index * self.batch_size: (index + 1) * self.batch_size
                ],
                self.y: self.train_set_y[
                    index * self.batch_size: (index + 1) * self.batch_size
                ]
            }
        )


        train_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: self.train_set_x[
                    index * self.batch_size: (index + 1) * self.batch_size
                ],
                self.y: self.train_set_y[
                    index * self.batch_size: (index + 1) * self.batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: self.test_set_x[
                    index * self.batch_size: (index + 1) * self.batch_size
                ],
                self.y: self.test_set_y[
                    index * self.batch_size: (index + 1) * self.batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: self.valid_set_x[
                    index * self.batch_size: (index + 1) * self.batch_size
                ],
                self.y: self.valid_set_y[
                    index * self.batch_size: (index + 1) * self.batch_size
                ]
            }
        )

        score_score_raw = theano.function(
            [],
            self.logLayer.p_y_given_x,
            givens={
                self.x: self.score_set_x
            }
        )

        score_score_outcome = theano.function(
            [],
            self.logLayer.y_pred,
            givens={
                self.x: self.score_set_x
            }
        )

        score_train_raw = theano.function(
            [],
            self.logLayer.p_y_given_x,
            givens={
                self.x: self.train_set_x
            }
        )

        score_train_outcome = theano.function(
            [],
            self.logLayer.y_pred,
            givens={
                self.x: self.train_set_x
            }
        )

        # Create a function that scans the entire training set
        def train_score():
            return [train_score_i(i) for i in xrange(num_train_batches)]

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        def make_predictions():
            return score_score_raw(), score_score_outcome(), score_train_raw(), score_train_outcome()

        return train_fn, train_score, valid_score, test_score, make_predictions

    def save_predictions_raw(self):

        if type(self.raw_predictions) != numpy.ndarray:
            print 'Train the model before saving predictions!!'
        else:
            print 'saving file ' + 'DBN_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '_' + 'pred.csv'
            numpy.savetxt('DBN_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '_' + 'pred_raw.csv', self.raw_predictions, delimiter=",")

    def save_predictions_outcome(self):

        if type(self.outcome_predictions) != numpy.ndarray:
            print 'Train the model before saving predictions!!'
        else:
            print 'saving file ' + 'DBN_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '_' + 'pred.csv'
            numpy.savetxt('DBN_' + self.pickle_dataset.strip('.gz').strip('.pkl') + '_' + str(self.rand_seed) + '_' + 'pred_outcome.csv', self.outcome_predictions, delimiter=",")

    def train(self):
        """
        Demonstrates how to train and test a Deep Belief Network.

        This is demonstrated on MNIST.

        :type finetune_lr: float
        :param finetune_lr: learning rate used in the finetune stage
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining
        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        :type k: int
        :param k: number of Gibbs steps in CD/PCD
        :type training_epochs: int
        :param training_epochs: maximal number of iterations ot run the optimizer
        :type dataset: string
        :param dataset: path the the pickled dataset
        :type batch_size: int
        :param batch_size: the size of a minibatch
        """

        # compute number of minibatches for training, validation and testing
        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / self.batch_size

        # numpy random generator
        print '... building the model'
        # construct the Deep Belief Network

        # start-snippet-2
        #########################
        # PRETRAINING THE MODEL #
        #########################
        print '... getting the pretraining functions'
        pretraining_fns = self.pretraining_functions()

        print '... pre-training the model'
        start_time = timeit.default_timer()
        ## Pre-train layer-wise
        for i in xrange(self.n_layers):
            # go through pretraining epochs
            for epoch in xrange(self.pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=self.pretrain_lr))
                if self.verbose:
                    print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                    print numpy.mean(c)

        end_time = timeit.default_timer()
        # end-snippet-2
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, train_model, validate_model, test_model, score_model = self.build_finetune_functions()

        print '... finetuning the model'
        # early-stopping parameters
        patience = self.patience * n_train_batches  # look as this many examples regardless
        patience_increase = float(self.patience_increase)    # wait this much longer when a new best is
                                  # found
        improvement_threshold = 0.9995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatches before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        self.raw_predictions = None
        self.outcome_predictions = None
        self.raw_predictions_train = None
        self.outcome_predictions_train = None

        while (epoch < self.training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:

                    train_losses = train_model()
                    this_training_loss = numpy.mean(train_losses)

                    test_losses = test_model()
                    this_testing_loss = numpy.mean(test_losses)

                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses)

                    if self.verbose:

                        print(
                            'epoch {}, minibatch {}/{}, training error {:.2f} %, validation error {:.2f} %, testing error {:.2f} %'.format(
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                this_training_loss * 100.,
                                this_validation_loss * 100.,
                                this_testing_loss * 100.
                            )
                        )

                    # if we got the best validation score until now

                    train_gate = True
                    valid_gate = (best_validation_loss - this_validation_loss > 0.01) or ((abs(this_validation_loss - best_validation_loss) < .0000001) and (this_training_loss < corr_training_loss))
                    if self.enforce_train_supremacy:
                        train_gate = this_training_loss < this_validation_loss

                    if valid_gate and train_gate:
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        corr_testing_loss = this_testing_loss
                        best_validation_loss = this_validation_loss
                        corr_training_loss = this_training_loss
                        best_epoch = epoch
                        self.raw_predictions, self.outcome_predictions, self.raw_predictions_train, self.outcome_predictions_train = score_model()
                        best_iter = iter


                        # for element in dbn.params:
                        #     print element.get_value()
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               corr_testing_loss * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %.2f %%, '
                'obtained at epoch %i iteration %i, '
                'with test performance %.2f %% '
                'with training performance %.2f %% '
            ) % (best_validation_loss * 100., best_epoch, best_iter + 1, corr_testing_loss * 100., corr_training_loss * 100.)
        )
        print >> sys.stderr, ('The fine tuning code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time)
                                                  / 60.))


if __name__ == '__main__':
    DBN_fun = DBN(verbose=True, rand_seed=45)
    DBN_fun.train()
    ensemble = DBN_fun.raw_predictions
    DBN_fun.save_predictions_raw()
    DBN_fun.save_predictions_outcome()
    #
    # count = 1.0
    #
    # for rand_seed_test in range(1,11):
    #     count += 1.0
    #     DBN_fun = DBN(rand_seed=rand_seed_test,verbose=False)
    #     DBN_fun.train()
    #     ensemble += DBN_fun.raw_predictions
    #     print count
    #
    # ensemble = ensemble / count
    #
    # numpy.savetxt('numerai_ensemble.csv', ensemble, delimiter=",")
