import numpy as np

def onehot_encode(integer_labels, n_labels):
    ''' 
    integer_labels is a numpy vector of integers
    Return a matrix where each row is a onehot vector.
    ''' 
    if len(integer_labels.shape) != 1: raise ValueError('integer_labels should be a numpy vector')

    size = len(integer_labels)
    onehot = np.zeros((size, n_labels), dtype='uint8')
    onehot[np.arange(size), integer_labels] = 1
    return onehot
