import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    '''
    Data generator
    '''
    def __init__(self, x_set, y_set, batch_size=1, augment=None):
        '''
        Initialization
        '''
        assert(len(x_set)==len(y_set))
        assert(batch_size > 0 and batch_size < len(x_set))

        self._x_set = x_set
        self._y_set = y_set
        self._augment = augment
        self._batch_size = batch_size

        self._indices = np.arange(len(x_set))
        np.random.shuffle(self._indices)
        
    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        return int(np.ceil(len(self._x_set) / self._batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''
        start_idx = index * self._batch_size
        end_idx = (index + 1) * self._batch_size 
        end_idx = end_idx if end_idx < len(self._x_set) else len(self._x_set)

        indices = self._indices[start_idx:end_idx]
        x_set = None 
        if self._augment: 
            x_set = np.take(self._x_set, indices, axis=0)
            x_set = self._augment(x_set)
        else:
            x_set = np.take(self._x_set, indices, axis=0)
        return (x_set, np.take(self._y_set, indices, axis=0))

    @property
    def batch_size(self):
        '''
        Gets a new batch size
        '''
        return self._batch_size 

    @batch_size.setter
    def batch_size(self, batch_size):
        '''
        Sets a new batch size
        '''
        if batch_size < 1: raise ValueError("batch_size < 1")
        self._batch_size = batch_size

    @property
    def x(self):
        '''
        Returns all x values
        '''
        return self._x_set

    @property
    def y(self):
        '''
        Returns all y values
        '''
        return self._y_set
    
    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        '''
        np.random.shuffle(self._indices)
