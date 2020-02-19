import os
from cifar_helpers import cifar10_binaries_to_tensor
from common.datasets import DatasetH5, write_h5_dataset

# We process binary into a h5 format compatible with DatasetH5, so that it the same architecture can be used over different datasets
# both on the farm and on our workstations
SOURCE_PATH = '/data/raw/cifar-10-batches-bin'
OUTPUT_PATH = '/data/cifar10'
    
TRAINING_FILES = ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin', 'data_batch_4.bin', 'data_batch_5.bin']
TESTING_FILES = ['test_batch.bin']

if __name__ == '__main__':
    # make the directory if it does not exist
    if not os.path.isdir(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)

    print('Processing training data')
    x_set, y_set = cifar10_binaries_to_tensor(SOURCE_PATH, TRAINING_FILES)

    train_filename = os.path.join(OUTPUT_PATH, DatasetH5.train_file_prefix + '_00001' + DatasetH5.ext)
    write_h5_dataset(train_filename, x_set, y_set)

    # load up all test data
    print('Processing testing data')
    x_set, y_set = cifar10_binaries_to_tensor(SOURCE_PATH, TESTING_FILES) 
    test_filename = os.path.join(OUTPUT_PATH, DatasetH5.test_file_prefix + '_00001' + DatasetH5.ext)
    write_h5_dataset(test_filename, x_set, y_set)
    
    # Checking that it works
    print('Checking that dataset loads')
    cifar10 = DatasetH5(OUTPUT_PATH)
    
    print('Done')
