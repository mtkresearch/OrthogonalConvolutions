import numpy as np
from common.datasets import *

# Checks that the cifar 10 dataset specified can be read. Compares some loaded data to that which has been previously verified.
CIFAR10_PATH = '/data/cifar10'
SEED = 1
N_TRAIN = 50000
N_TEST = 10000
N_TRAIN_SAMPLES = 500
N_TEST_SAMPLES = 100

if __name__ == '__main__':
    # Get indices of samples
    np.random.seed(SEED)
    indices_train = np.random.randint(N_TRAIN, size=N_TRAIN_SAMPLES); 
    indices_test = np.random.randint(N_TEST, size=N_TEST_SAMPLES); 

    print('Load data')
    cifar10 = DatasetH5(CIFAR10_PATH)

    print('Gather samples')
    # get samples from generator
    train_gen = cifar10.generator_train_data
    test_gen = cifar10.generator_test_data

    train_samples_x = train_gen.x[indices_train]
    train_samples_y = train_gen.y[indices_train]
    test_samples_x = test_gen.x[indices_test]
    test_samples_y = test_gen.y[indices_test]

    print('Load saved samples')
    # samples from pkl file 
    with open('cifar10_samples.pkl', 'rb') as load_file:
        train_samples_x_load = np.load(load_file)
        train_samples_y_load = np.load(load_file)
        test_samples_x_load = np.load(load_file)
        test_samples_y_load = np.load(load_file)

    # check that they match
    assert(np.array_equal(train_samples_x, train_samples_x_load))
    assert(np.array_equal(train_samples_y, train_samples_y_load))
    assert(np.array_equal(test_samples_x, test_samples_x_load))
    assert(np.array_equal(test_samples_y, test_samples_y_load))

    print('Cifar10 check done!')
