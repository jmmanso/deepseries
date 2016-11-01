""" Auxiliary tools for data pre-processing.
Includes the following classes:

- `BatchSampler`, iterator for splitting
	arrays into batches
- `Slicer`, slices arrays into chunks of
	fixed size
- `Balancer`, given an array of class labels,
	returns an array of weights that balances
	the classes
- `Datetime2phase`, transforms a list of datetimes
    into a multi-channel phase array
"""


import numpy as np
import pandas as pd
import datetime




class BatchSampler:
    ''' Takes a list of arrays of the same
    dimension along the 0-axis. Randomizes
    their positions, and applies an iterator
    that returns a batch list akin to the input
     '''

    def __init__(self,
                 array_list,
                 batch_size=50):

        # check that arrays have the same length
        # alox the 0-axis
        self.array_list = array_list
        self.check_arrays(self.array_list)
        # determine the length of the arrays with
        # the first object in the list
        self.full_size = len(self.array_list[0])
        # define batch dimensions
        self.batch_size = batch_size
        self.Nbatches = self.full_size // self.batch_size
        # define index grid and randomize
        self.random_indices = np.arange(self.full_size)
        np.random.shuffle(self.random_indices)
        # inititialize counter
        self.iter_counter = 0

    @staticmethod
    def check_arrays(array_list):
        array_types = [type(a).__name__ for a in array_list]
        array_lengths = [np.array(a).shape[0] for a in array_list]
        if len(set(array_types)) > 1:
            raise IOError('Please input arrays of the same type')
        #
        if len(set(array_lengths)) > 1:
            raise IOError('Please input arrays of the same length')

    def next(self):
        if self.iter_counter >= self.Nbatches:
            raise StopIteration
        else:
            idx_selection = self.random_indices[self.iter_counter *
                                                self.batch_size:(self.iter_counter + 1) * self.batch_size]
            #
            batch_list = [a[idx_selection] for a in self.array_list]
            self.iter_counter += 1
            return batch_list

    def __iter__(self):
        return self


class Slicer(object):

    def fit(self,
            data_explanatory,
            data_response=None,
            sequence_length=10,
            sequence_offset=0,
            kind='seq2seq'):

        self.data_explanatory = self.format(data_explanatory)
        if type(data_response).__name__ == 'NoneType':
            self.data_response = self.data_explanatory
        else:
            self.data_response = self.format(data_response)

        self.Ne, self.Ke = self.data_explanatory.shape
        self.Nr, self.Kr = self.data_response.shape
        self.kind = kind
        self.sequence_length = sequence_length
        self.sequence_offset = sequence_offset
        self.sanity_checks()
        return self.slice_it()

    def sanity_checks(self):
        assert(self.Ne == self.Nr), \
            "Explanatory and response data need to have the same length"
        assert self.kind in ['seq2seq', 'seq2num'], \
            "Kind must be seq2seq or seq2num"
        assert self.sequence_length < self.Ne, \
            "Sequence length must be smaller than dataset length"

    def slice_it(self):

        # Sequences are not independent. Sucessive sequences
        # are the same but shifted one element to the future.
        #
        self.N_sequences = self.Ne - self.sequence_length - self.sequence_offset
        self.container_explanatory = []
        self.container_response = []

        for i in range(self.N_sequences):
            self.container_explanatory.append(
                self.data_explanatory[i:i + self.sequence_length])
            if self.kind == 'seq2seq':
                self.container_response.append(
                    self.data_response[i + 1 + self.sequence_offset:i + 1 + self.sequence_length + self.sequence_offset])
            elif self.kind == 'seq2num':
                self.container_response.append(
                    [self.data_response[i + self.sequence_length + self.sequence_offset]])
            else:
                raise TypeError('Invalid kind')
        #
        self.container_explanatory = np.array(
            self.container_explanatory, dtype=np.float64)
        self.container_response = np.array(
            self.container_response, dtype=np.float64)
        return self.container_explanatory, self.container_response

    def format(self, data):
        ''' Transforms a one-dimensional structure
        into a numpy array. Inputs can be numpy array,
        list, tuple, dataframe, series. '''
        obj_type = type(data).__name__
        if obj_type in ['DataFrame', 'Series']:
            data = data.values
        elif obj_type == 'ndarray':
            data = data
        elif obj_type in ['list', 'tuple']:
            data = np.array(data)
        else:
            raise IOError(
                'Input data not recognized. Should be ndarray, list, tuple, Series or DataFrame.')
        #
        # If shape is (N,), convert to (N,1):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return 1.0 * data


class Balancer(object):
    ''' Class balancer '''

    def fit(self, label_array):
        ''' Takes a 1-D array of integer labels,
        returns a weighted array of shame shape. The
        weight sum is the same for each class, and should
        result in 1/n_classes. The total
        weight sum across all classes is 1. '''
        label_array = self.check_array(label_array)
        label_set = set(label_array)
        array_length = len(label_array)
        set_length = len(label_set)
        label_list = label_array.tolist()
        count_dict = {i: label_list.count(i) for i in label_set}
        norm_dict = {i: 1 - count_dict[i] *
                     1.0 / array_length for i in label_set}
        weight_array = np.array([norm_dict[j]
                                 for j in label_array], dtype=np.float64)
        normed_weight_array = weight_array / sum(weight_array)
        return normed_weight_array

    def check_array(self, array):
        ''' Make sure array is 1D and int type '''
        array = np.array(array, dtype=np.int32).ravel()
        return array





class Datetime2phase:
    """
    Transforms a datetime list into a multi-channel
    array of year, month, day, etc. The idea is to make
    time phase explicit for every relevant timescale. Each
    channel carries a float between [0,1].
    """
    def __init__(self, dt_list):
        self.time_columns = ['year', 'month', 'day',
                'weekday', 'hour', 'minute',
                'second']
        self.transformations = {'year': lambda x: (x-2000)/100.,
                        'month': lambda x: (x-1.0)/12.,
                        'day': lambda x: (x-1.0)/30.5,
                        'weekday':lambda x: x/7.,
                        'hour': lambda x: x/24.,
                        'minute': lambda x: x/60.,
                        'second': lambda x: x/60.}
        self.time_frame = self.formatter(dt_list)

    def formatter(self, dt_list):
        """
        takes a datetime object list
        """
        tup_cont = []
        for dt in dt_list:
            tup = (dt.year,
                dt.month,
                dt.day,
                dt.weekday(),
                dt.hour,
                dt.minute,
                dt.second)
            tup_cont.append(tup)
        s = pd.DataFrame(tup_cont,
            columns=self.time_columns)
        return s

    def normalizer(self, df_):
            df = df_.copy()
            for timescale, t in self.transformations.items():
                df[timescale] = df[timescale].apply(t)
            #
            return df

    def __call__(self):
        return self.normalizer(self.time_frame)




















#
