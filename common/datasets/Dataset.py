from abc import ABC, abstractmethod
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .DataGenerator import DataGenerator
import h5py
import os
import numpy as np

def write_h5_dataset(filename, x_set, y_set):
    ''' 
    Writes a dataset as a h5 file.  Dataset_tuple has the folThis is an helper function used to dump datasets in h5 format.
    ''' 
    with h5py.File(filename, 'w') as h5file:
        group = h5file.create_group('dataset')
        group.create_dataset('x_set', data=x_set, compression='gzip', compression_opts=9)
        group.create_dataset('y_set', data=y_set, compression='gzip', compression_opts=9)

def read_h5_dataset(filename):
    ''' 
    Reads a h5 dataset. and returns it
    ''' 
    with h5py.File(filename, 'r') as h5File:
        if 'dataset' not in h5File.keys(): raise ValueError('h5 file has no dataset')
        dataset = h5File['dataset']
        x_set = np.array(dataset['x_set'])
        y_set = np.array(dataset['y_set'])

    return x_set, y_set

class Dataset(ABC):
    @property
    @abstractmethod
    def input_shape(self):
        '''
        returns the input shape (tuple)
        '''
        pass

    @property
    @abstractmethod
    def output_shape(self):
        '''
        returns the output shape (tuple)
        '''
        pass

    @property
    @abstractmethod
    def n_train(self):
        '''
        returns the number of training samples
        '''
        pass

    @property
    @abstractmethod
    def n_test(self):
        '''
        returns the number of testing samples
        '''
        pass

    @property
    @abstractmethod
    def generator_train_data(self):
        '''
        Training data generator 
        '''
        pass

    @property
    @abstractmethod
    def generator_test_data(self):
        '''
        Testing data generator 
        '''
        pass

class DatasetGenerators(Dataset):
    def __init__(self, generator_train_data, generator_test_data):
        self._generator_train_data = generator_train_data
        self._generator_test_data = generator_test_data
        self._input_shape = generator_train_data[0][0].shape[1:]
        self._output_shape = generator_train_data[0][1].shape[1:]
        self._n_train = len(generator_train_data)
        self._n_test = len(generator_test_data)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def n_train(self):
        return self._n_train

    @property
    def n_test(self):
        return self._n_test

    @property
    def generator_train_data(self):
        return self._generator_train_data

    @property
    def generator_test_data(self):
        return self._generator_test_data

# Class to handle image datasets using keras ImageDataGenerators
class DatasetImagesH5(DatasetGenerators):
    train_file_prefix = 'train'
    test_file_prefix = 'test'
    ext = '.h5'

    def __init__(self, path, augment=None):
        '''
        Initialize dataset from h5 file 
        '''
        self._path = path
        self._augment = augment
        self._train_filename = os.path.join(path, DatasetImagesH5.train_file_prefix + '_00001' + DatasetImagesH5.ext)
        self._test_filename = os.path.join(path, DatasetImagesH5.test_file_prefix + '_00001' + DatasetImagesH5.ext)

        # TODO(alvin 20191216): Consider producing tf.data.Dataset rather than Data Generators
        # We are sticking to this for now in view of having more control over larger data sets, but it
        # is something that we should consider supporting as well 
        x_set, y_set = read_h5_dataset(self._train_filename)
        train_data_generator = ImageDataGenerator(**self._augment) if self._augment else ImageDataGenerator()
        train_data_generator.fit(x_set)
        train_data_generator = train_data_generator.flow(x_set, y_set, batch_size=1)

        x_set, y_set = read_h5_dataset(self._test_filename)
        test_data_generator = ImageDataGenerator()
        test_data_generator.fit(x_set)
        test_data_generator = test_data_generator.flow(x_set, y_set, batch_size=1)

        super().__init__(train_data_generator, test_data_generator)

