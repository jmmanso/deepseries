""" Time series prediction based on LSTMs.

There are 4 clases in this module:

- ``DeepSeries`` base class, models sequence to sequence LSTMs
- ``Sequencer`` meta class that uses ``DeepSeries``
  for sequence-to-sequence prediction, supporting classification
  and regression and arbitrary number of input & output time steps
- ``many2oneClassifier``, meta class that uses ``DeepSeries``
  for sequence-to-label classification
- ``many2oneRegressor``, meta class that uses ``DeepSeries``
  for sequence-to-number regression

"""

# Author: Jesus Martinez-Manso (jesus.martinezmanso@capitalone.com)


import numpy as np
import inspect
import tensorflow as tf
from abc import ABCMeta
import six
import types
import warnings
# other modules from this package:
import preprocessing
import functions


class DeepSeries(object):
    """Base class for sequence prediction. Supports:
            - classification and regression
            - custom normalization weights for training data
            - learns from sequences of fixed length,
              predicts sequences of any length

    Warnings:
            (1) This class is not meant to be used directly. Use derived classes
            instead.

        (2) The following argument names used in the TensorFlow API have the following
        names here:
        ``batch_size`` -> ``Nsequences``
        ``max_time`` -> ``sequence_length``
        ``input_size`` -> ``Nchannels``

    Parameters
    ----------
    train_X : numpy array of shape = (Nsequences, sequence_length, Nchannels)
        The training input sequences.

    train_y : numpy array of shape = (Nsequences, sequence_length, Nchannels)
        The training target sequences.

    test_X : numpy array of shape = (Nsequences, sequence_length, Nchannels)
        The test input sequences. Used only to evaluate loss function and compare
        to training.

    test_y : numpy array of shape = (Nsequences, sequence_length, Nchannels)
        The test target sequences. Used only to evaluate loss function and compare
        to training.

    NOTE: train_X, train_y, test_X and test_Y do NOT need to have the same shape.
    Variations in each dimension might be allowed depending on the particular application.

    y_categorical_indices_container: None, 'all' or list (default=None)
        Indicates the indices in train_X that represent one-hot encoded
        categorical features. `None` means that all features should be treated as
        continuous, 'all' means that all features are categorical, an empty list
        [] is equivalent to None, and an index list simply picks those columns,
        such as [0,1,5] would determine train_X[:,[0,1,5]] as categorical and all
        remaining columns as continuous

    train_norm_coeff : numpy array of shape = (Nsequences,), optional (default=None)
            Normalization weights for the training samples. The norm of this array
            will later be set to the number of samples.

    test_norm_coeff : numpy array of shape = (Nsequences,), optional (default=None)
            Same as train_norm_coeff, but used to normalize the test set.

    Nnodes : integer, (default=20)
            Number of nodes in the LSTM cell

    Nlayers : integer, (default=1)
            Number of cell layers. Cells can be stacked.

    input_keep_prob : float between 0 and 1, (default=1.0)
            Probabilty to keep inputs. Enter values less than 1 to perform dropout.
            IMPORTANT: This applies to training AND evaluation, thus several predictions
            on the same sample will differ (slightly) if input_keep_prob<1 or input_keep_prob<1.

    output_keep_prob : float between 0 and 1, (default=1.0)
            Probabilty to keep outputs. Enter values less than 1 to perform dropout.

    optimizer_type : string, name of tensorflow.train optimizer (default='AdamOptimizer')

    optimizer_kwargs : dictionary, arguments to be passed to the
            optimizer (default={'learning_rate':0.001})

    cell_type : string, name of tensorflow.nn.rnn_cell cell class, (default='LSTMCell')

    cell_kwargs : dictionary, arguments to be passed to the cell
            (default={'use_peepholes':False, 'forget_bias':1,'state_is_tuple':True})
            If any of the keys are not arguments of the cell it is being fed to, they will
            just be ignored without causing an error

    batch_size : integer, (default=50)
            Number of samples to include in each batch during SGD optimization

    tracking_step : integer, (default=20)
            As the optimization iterates over batches, at every ``tracking_step`` it
            will print out the train loss value for the current batch. It will also
            compute the loss value of the entire test sample and print it as well.
    """

    def __init__(self,
                 train_X,
                 train_y,
                 test_X,
                 test_y,
                 y_categorical_indices_container=None,
                 train_norm_coeff=None,
                 test_norm_coeff=None,
                 Nnodes=20,
                 Nlayers=1,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 optimizer_type='AdamOptimizer',
                 optimizer_kwargs={'learning_rate': 0.001},
                 cell_type='LSTMCell',
                 cell_kwargs={'use_peepholes': False,
                              'forget_bias': 1, 'state_is_tuple': True},
                 batch_size=50,
                 tracking_step=20):
        # input arrays
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        # Check input array shapes
        self.Nsequences, self.sequence_length, self.Nchannels = self.train_X.shape
        self.Nsequences_target, self.sequence_length_target, self.Nchannels_target = \
            self.train_y.shape
        assert(self.Nsequences ==
               self.Nsequences_target), "Explanatory and response data need to have same length"

        # if no categorical indices are passed, convert to empty list
        if type(y_categorical_indices_container) == types.NoneType:
            self.y_categorical_indices_container = []
        elif type(y_categorical_indices_container) in [types.TupleType,types.ListType]:
            self.y_categorical_indices_container = y_categorical_indices_container
        elif type(y_categorical_indices_container) == types.StringType:
            if y_categorical_indices_container == 'all':
                self.y_categorical_indices_container = [range(self.Nchannels_target)]
            else:
                raise Exception("The only valid str value for y_categorical_indices_container is 'all'")
        #
        self.all_y_categorical_indices = np.ravel(self.y_categorical_indices_container)
        # build a list of continuous indices, which can be empty
        self.y_continuous_indices = \
            [i for i in range(self.Nchannels_target) if i not in self.all_y_categorical_indices]





        # norm coefficients. If None are passed, set to arrays of ones.
        # Normalize these arrays to Nsequences.
        if type(train_norm_coeff).__name__ == 'NoneType':
            self.train_norm_coeff = np.ones(len(self.train_y))
        else:
            self.train_norm_coeff = np.array(
                train_norm_coeff) * len(self.train_y) / sum(train_norm_coeff)

        if type(test_norm_coeff).__name__ == 'NoneType':
            self.test_norm_coeff = np.ones(len(self.test_y))
        else:
            self.test_norm_coeff = np.array(
                test_norm_coeff) * len(self.test_y) / sum(test_norm_coeff)

        assert(len(self.train_norm_coeff) == len(self.train_y)
               ), "Target and norm coefficients must have same shape"
        assert(len(self.test_norm_coeff) == len(self.test_y)
               ), "Target and norm coefficients must have same shape"
        # reshape to (N,1)
        self.train_norm_coeff = self.train_norm_coeff.reshape(-1, 1)
        self.test_norm_coeff = self.test_norm_coeff.reshape(-1, 1)
        # numerical params
        self.Nnodes = Nnodes
        self.batch_size = batch_size
        self.Nbatches = self.Nsequences // self.batch_size
        self.tracking_step = tracking_step
        self.Nlayers = Nlayers
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        # categorical params
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        # get cell and optimization objects
        self.cell = getattr(tf.nn.rnn_cell, cell_type)
        self.optimizer = getattr(tf.train, self.optimizer_type)
        # keep only those args that are accepted by the cell
        self.cell_kwargs = {key: cell_kwargs[key] for key in cell_kwargs.keys()
                            if key in inspect.getargspec(self.cell.__init__).args}
        # start graph and session
        self.init_graph()

    def make_variables(self):
        """Define variables within the TF graph"""
        # input sample of sequences
        self.X = tf.placeholder(
            tf.float32, [None, self.sequence_length, self.Nchannels])
        self.y = tf.placeholder(
            tf.float32, [None, self.sequence_length_target, self.Nchannels_target])
        # slice each time step from the target sequences, and wrap it in a list
        self.y_sequence = [self.y[:, i, :]
                           for i in range(self.sequence_length_target)]
        # output params
        self.weights = tf.Variable(tf.truncated_normal(
            [self.Nnodes, self.Nchannels_target]), trainable=True)
        self.biases = tf.Variable(tf.constant(
            0.1, shape=[self.Nchannels_target]), trainable=True)
        # normalization coefficients
        self.norm_coeff = tf.placeholder(tf.float32, [None, 1])
        # Container lists to append loss values during fitting. These are not
        # graph variables.
        self.train_cost_container = []
        self.test_cost_container = []

    def make_cell(self):
        """ Build the RNN objects """
        # Instantiate base cell
        self.base_cell = self.cell(self.Nnodes, **self.cell_kwargs)
        # Create cell stack
        self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.base_cell] * self.Nlayers,
                                                        state_is_tuple=True)
        # Add dropout
        self.stacked_cell = tf.nn.rnn_cell.DropoutWrapper(self.stacked_cell,
                                                          input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
        # define output and state vectors
        self.output, self.states = tf.nn.dynamic_rnn(
            self.stacked_cell, self.X, dtype=tf.float32)
        # slice each time step from the target sequences. Keep only the last
        # ``sequence_length_target`` elements
        self.output_slices = [self.output[:, i, :]
                              for i in range(self.sequence_length - self.sequence_length_target, self.sequence_length)]
        # Also store the state of the last layer. The self.state variable is a list of Nlayers,
        # each one being a tensor of shape (None[n_sequences], n_nodes)
        self.final_state = self.states[-1]

    def make_ops(self):
        """ Build graph operations """
        # output_logits is a list of ``self.sequence_length_target`` elements, one
        # logit for each unrolled time step
        self.output_logits = [tf.matmul(slice_, self.weights) + self.biases
                              for slice_ in self.output_slices]
        # get the the cost function
        self.cost_function = functions.cost_functions.combined
        # assing tensors to cost function and define the operation. Essentially,
        # we compute the cost for each element, multiply by the norm coefficient,
        # and sum over all elements

        self.cost_operation = tf.reduce_sum(
                [self.cost_function(logits,
                    t,
                    self.y_categorical_indices_container,
                    self.y_continuous_indices,
                    self.norm_coeff)
                 for logits, t in zip(self.output_logits,
                                      self.y_sequence)])

        # Define operation minimizer
        self.minimizer = self.optimizer(
            **self.optimizer_kwargs).minimize(self.cost_operation)
        # get the prediction function
        self.predict_function = functions.prediction_functions.combined
        # This operation will compute the prediction for the unrolled logits
        self.predict_operation = [self.predict_function(
                                    logits,
                                    self.y_categorical_indices_container,
                                    self.y_continuous_indices)
                                for logits in self.output_logits]

    def init_graph(self):
        """ Initialize graph """
        # close any open session
        self.close()
        # reset graph
        tf.reset_default_graph()
        # define variables and operations
        self.make_variables()
        self.make_cell()
        self.make_ops()
        # initialize all variables
        init_op = tf.initialize_all_variables()
        # start new session
        self.sess = tf.Session()
        self.sess.run(init_op)

    def close(self):
        ''' Closes any open session '''
        if hasattr(self, 'sess'):
            self.sess.close()

    def fit_single_epoch(self, verbose=True):
        """ Runs one fitting epoch """
        # Slice the input samples into batches
        bs_train = preprocessing.BatchSampler([self.train_X, self.train_y, self.train_norm_coeff],
                                              batch_size=self.batch_size)
        # Iterate over batches
        for i, (batch_x_train, batch_y_train, batch_w_train) in enumerate(bs_train):
            # Minimize loss and learn
            self.sess.run(self.minimizer,
                          feed_dict={self.X: batch_x_train, self.y: batch_y_train, self.norm_coeff: batch_w_train})
            # Print loss progress
            if i % self.tracking_step == 0:
                # Compute training batch cost
                this_train_cost = self.sess.run(self.cost_operation,
                                                feed_dict={self.X: batch_x_train, self.y: batch_y_train, self.norm_coeff: batch_w_train}) / self.batch_size
                # Compute full test cost
                this_test_cost = self.sess.run(self.cost_operation,
                                               feed_dict={self.X: self.test_X, self.y: self.test_y, self.norm_coeff: self.test_norm_coeff}) / len(self.test_y)
                display_message = 'train batch cost: %1.3f, test cost: %1.3f ' % (
                    this_train_cost, this_test_cost)
                # Append costs to containers
                self.train_cost_container.append(this_train_cost)
                self.test_cost_container.append(this_test_cost)
                # Print progress
                if verbose:
                    print display_message

    def fit(self, n_epochs=1, verbose=True):
        ''' Iterates fitting over a number of epochs '''
        for _ in range(n_epochs):
            self.fit_single_epoch(verbose=verbose)

    def base_unroll(self, X, return_softmax=True):
        """ Base method for sequence unrolling.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
                The input samples.

        return_softmax : boolean
                Whether the output for categorical variables should
                be returned as softmax probabilities. If True and there
                are no categorical variables, then it takes no effect

        Returns
        -------
        y : numpy array, shape = (Nsequences, sequence_length_output, Nchannels_ouput)
            The predicted outputs.  """

        y = self.sess.run(self.predict_operation, feed_dict={self.X: X})
        y = np.array(y)
        # y is array of shape (sequence_length, Nsequences, Nchannels),
        # we convert it here to (Nsequences, sequence_length, Nchannels):
        y = np.transpose(y, axes=[1, 0, 2])
        # If we want to return one-hot states for the categorical variables
        if not return_softmax and len(self.all_y_categorical_indices)>0:
            y[:,:,self.all_y_categorical_indices] = \
                functions.softmax2onehot(y[:,:,self.all_y_categorical_indices])

        return y

    def get_state(self, X):
        """ Returns final state vectors for an array of inputs """
        states = self.sess.run(self.final_state, feed_dict={self.X: X})
        return states


class Sequencer(six.with_metaclass(ABCMeta, DeepSeries)):

    def __init__(self,
                 train_X,
                 train_y,
                 test_X,
                 test_y,
                 y_categorical_indices_container=None,
                 train_norm_coeff=None,
                 test_norm_coeff=None,
                 Nnodes=60,
                 Nlayers=1,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 optimizer_type='AdamOptimizer',
                 optimizer_kwargs={'learning_rate': 0.001},
                 cell_type='LSTMCell',
                 cell_kwargs={'use_peepholes': False, 'forget_bias': 1,
                                'state_is_tuple': True},
                 batch_size=50,
                 tracking_step=20):
        super(Sequencer, self).__init__(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            y_categorical_indices_container=y_categorical_indices_container,
            train_norm_coeff=train_norm_coeff,
            test_norm_coeff=test_norm_coeff,
            Nnodes=Nnodes,
            Nlayers=Nlayers,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            cell_type=cell_type,
            cell_kwargs=cell_kwargs,
            batch_size=batch_size,
            tracking_step=tracking_step)
        # confirm that it is self-sequencing
        assert(self.Nchannels_target == self.Nchannels), \
            "self-sequencing requires input and output data to be of same kind and shape"

    def unroll(self, X, n_output=1, flatten=False):
        """ Unrolls future time steps based on input sequence.
        Requires explanatory and response data to be the same.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
                The input samples.

        n_output : integer, (default=1)
                Number of time steps to predict. When larger than one, the
                network will use its previous output as new input, in a
                rolling-window mode. The window is of size ``self.sequence_length``.

        flatten : boolean, (default=False)
                Whether output array should be flattened. Useful when
                Nsequences=1 and Nchannels=1.

        Returns
        -------
        X_output : numpy array, shape = (Nsequences, n_output, Nchannels_ouput)
            The predicted outputs. Only contains predictions unrolled past
            the input sequence """

        assert(len(X.shape) ==
               3), "Input array has wrong number of dimensions, should be (N,T,K)"
        # Get X's dimensions
        N, T, K = X.shape
        if n_output > 1:
            # make sure the graph's input&output channels have same shape
            assert(self.Nchannels_target == self.Nchannels), "To use n_output>1, \
			the graph's input & output channels should be the same"

        # Different scenarios:
        # 1. Input sequence is incomplete
        #	 	1A. ...and we just want to complete it (n_output<=(sequence_length-T))
        #			Compete sequence gap.
        #
        # 	 	1B. ...and we want to predict beyond it.
        #	 		Complete sequence gap. Store. Compute (3), and concatenate
        #	 		results.
        #
        # 2. Input sequence is complete.
        #
        if T < self.sequence_length:
            if n_output <= self.sequence_length - T:
                X_output = self._complete_sequence(X, n_steps=n_output)
                # select only the predicted part of the sequence
                X_output = X_output[:, T:n_output + T, :]
            else:
                X_completed = self._complete_sequence(X)
                n_remaining = n_output - (self.sequence_length - T)
                X_extended = self._loop_sequences(
                    X_completed, n_output=n_remaining)
                X_output = np.hstack((X_completed, X_extended))
                # Return only the last n_output steps
                X_output = X_output[:, -n_output:, :]
        else:
            X_output = self._loop_sequences(X, n_output=n_output)
            X_output = X_output[:, -n_output:, :]

        if flatten:
            X_output = X_output.ravel()

        return X_output

    def _loop_sequences(self, X, n_output=1):
        """ Extends a sequence by predicting the next N elements.
        Requires explanatory and response data to be the same.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
                The input samples, where sequence_length is equal or greater than the instance
                native value

        n_output : integer, (default=1)
                Number of time steps to predict.

        Returns
        -------
        X_output : numpy array, shape = (Nsequences, n_output, Nchannels)
            The predicted sequence, excluding the input steps.
        """
        # ensure sequence_length is kosher
        assert(X.shape[1] >= self.sequence_length), \
            "Array sequence length %s should be larger than instance native value %s" \
            % (X.shape[1], self.sequence_length)
        # crop X to its last sequence_length, in case T is larger
        X = X[:, -self.sequence_length:, :]
        # make container for the output
        X_output = np.zeros((X.shape[0], n_output, X.shape[2]))
        # Start unrolling the predicted series
        for i in range(n_output):
            unrolling = self.base_unroll(X, return_softmax=False)
            #
            # construct the new X as the previous (minus its first element)
            # plus the last element of the unrolling.
            X = np.hstack((X[:, 1:, :], unrolling[:, -1:, :]))
            # add last new element to the output container
            X_output[:, i, :] = unrolling[:, -1, :]
        #
        return X_output

    def _complete_sequence(self, X, n_steps=None):
        """ Completes a sequence array that is missing time steps
        with respect to the instance's initialized sequence_length.
        Requires explanatory and response data to be the same.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
                The input samples, where sequence_length is less than the instance
                native value

        n_steps : integer, (default=None)
                Number of time steps to predict. When None, it will default
                to the steps left to complete the full sequence.

        Returns
        -------
        X_roll : numpy array, shape = (Nsequences, X.shape[1]+n_steps, Nchannels)
            The predicted sequence, including the input steps."""
        # get the array shape
        N, T, K = X.shape
        # if T (the array's input sequence length) is not smaller than the
        # native sequence length, there is no point to use this method.
        # Return X
        if T >= self.sequence_length:
            warnings.warn(
                'Sequence length in array is not smaller than the class native. Aborting.')
            return X[:, -self.sequence_length:, :]

        if not n_steps:
            # complete the entire sequence
            n_steps = self.sequence_length - T
        else:
            # n_steps should not exceed self.sequence_length - T
            n_steps = min(n_steps, self.sequence_length - T)
        # empty array to contain the predicted steps and the extra padding
        X_pad = np.zeros((N, self.sequence_length - T, K))
        # append to input array
        X = np.hstack((X, X_pad))
        X_roll = X.copy()
        # iteratively fill the first missing step.
        # If we could feed the cell output as input
        # in every unrolling, we could do all this in a
        # single unroll pass. However, TF RNN does not seem
        # to allow this, since it only accepts an array of fixed inputs.
        # Thus, in each pass, we gotta fetch the last predicted element
        # before the padded elements, and fill the array one at a time.
        for j in range(n_steps):
            unrolling = self.base_unroll(X_roll, return_softmax=False)
            X_roll[:, T + j:T + j + 1, :] = unrolling[:, T + j - 1:T + j, :]
        # return completed sequence up to n_steps
        return X_roll[:, :T + n_steps, :]


class many2oneClassifier(six.with_metaclass(ABCMeta, DeepSeries)):
    """ Meta class for prediction of numerical time series.
    See ``DeepSeries`` base class for a description of the parameters """

    def __init__(self,
                 train_X,
                 train_y,
                 test_X,
                 test_y,
                 train_norm_coeff=None,
                 test_norm_coeff=None,
                 Nnodes=60,
                 Nlayers=1,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 optimizer_type='AdamOptimizer',
                 optimizer_kwargs={'learning_rate': 0.001},
                 cell_type='LSTMCell',
                 cell_kwargs={'use_peepholes': False, 'forget_bias': 1,
                                'state_is_tuple': True},
                 batch_size=50,
                 tracking_step=20):
        super(many2oneClassifier, self).__init__(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            y_categorical_indices_container='all',
            train_norm_coeff=train_norm_coeff,
            test_norm_coeff=test_norm_coeff,
            Nnodes=Nnodes,
            Nlayers=Nlayers,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            cell_type=cell_type,
            cell_kwargs=cell_kwargs,
            batch_size=batch_size,
            tracking_step=tracking_step)

    def predict_proba(self, X, return_softmax=True):
        """ Prediction of class probabilities.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
                The input samples.

        return_softmax : boolean
                Whether the output for categorical variables should
                be returned as softmax probabilities.

        Returns
        -------
        y : numpy array, shape = (Nsequences, 1, Nchannels_ouput)
            The predicted output probabilities.
        """
        y = self.base_unroll(X, return_softmax=return_softmax)[:, -1:, :]
        return y

    def predict(self, X, return_class_indices=False):
        """ Prediction of classes.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
            The input samples.

        return_class_indices: Boolean, default=False
                Whether to return the predictions as a 1-D class value array,
                or to one-hot encode it

        Returns
        -------
        y : numpy array, shape = (Nsequences, 1, Nchannels)
        The predicted output classes as a one-hot array.
        """
        onehot_array = self.predict_proba(X, return_softmax=False)
        # get a reduction to the winning class index, useful when Nchannels==1
        best_class_indices = np.argmax(onehot_array, axis=2)
        if return_class_indices:
            return best_class_indices
        else:
            return onehot_array

    def classification_accuracy(self, use_weights=True):
        """Returns classification accuracy for the test sample.
        Uses weights if provided at initialization.
        """
        # output_logits is a list of tensors. Pack into single tensor, and
        # transpose so that the sample size is the first dimension
        self.output_logits_as_tensor = tf.transpose(
            tf.pack(self.output_logits), [1, 0, 2])
        # Now you can compare this tensor to the target data
        correct_prediction_ = tf.cast(tf.equal(tf.argmax(self.output_logits_as_tensor, 2),
                                               tf.argmax(self.y, 2)), "float")
        # Apply normalization coefficients
        if use_weights:
            correct_prediction = tf.mul(correct_prediction_, self.norm_coeff)
        else:
            correct_prediction = correct_prediction_
        # Calculate accuracy
        accuracy = tf.reduce_mean(correct_prediction)
        return self.sess.run(accuracy,
                             feed_dict={self.X: self.test_X, self.y: self.test_y, self.norm_coeff: self.test_norm_coeff})


class many2oneRegressor(six.with_metaclass(ABCMeta, DeepSeries)):
    """ Meta class for prediction of numerical time series.
    See ``DeepSeries`` base class for a description of the parameters """

    def __init__(self,
                 train_X,
                 train_y,
                 test_X,
                 test_y,
                 train_norm_coeff=None,
                 test_norm_coeff=None,
                 Nnodes=60,
                 Nlayers=1,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 optimizer_type='AdamOptimizer',
                 optimizer_kwargs={'learning_rate': 0.001},
                 cell_type='LSTMCell',
                 cell_kwargs={'use_peepholes': False, 'forget_bias': 1,
                                'state_is_tuple': True},
                 batch_size=50,
                 tracking_step=20):
        super(many2oneRegressor, self).__init__(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            y_categorical_indices_container=None,
            train_norm_coeff=train_norm_coeff,
            test_norm_coeff=test_norm_coeff,
            Nnodes=Nnodes,
            Nlayers=Nlayers,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            cell_type=cell_type,
            cell_kwargs=cell_kwargs,
            batch_size=batch_size,
            tracking_step=tracking_step)

    def predict(self, X, flatten=True):
        """ Prediction method.

        Parameters
        ----------
        X : numpy array, shape = (Nsequences, sequence_length, Nchannels)
                The input samples.
        flatten : boolean, (default=False)
                Whether output array should be flattened.

        Returns
        -------
        y : numpy array, shape = (Nsequences, 1, Nchannels_ouput)
            The predicted outputs.
        """
        y = self.base_unroll(X)[:, -1:, :]
        if flatten:
            y = y.ravel()
        return y
