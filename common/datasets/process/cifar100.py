import os
from cifar_helpers import cifar100_binaries_to_tensor
from common.datasets import DatasetH5, write_h5_dataset

# We process binary into a h5 format compatible with DatasetH5, so that it the same architecture can be used over different datasets
# both on the farm and on our workstations

SOURCE_PATH = '/data/raw/cifar-100-binary'
OUTPUT_PATH = '/data/cifar100'
CLASSIFY_FINE = True

TRAIN_FILENAME = 'train.bin'
TEST_FILENAME = 'test.bin'

N_TRAIN_IMAGES = 50000
N_TEST_IMAGES = 10000

N_LABELS_FINE = 100
N_LABELS_COARSE = 20

if __name__ == '__main__':
    n_labels = N_LABELS_FINE if CLASSIFY_FINE else N_LABELS_COARSE
    label_offset = 1 if CLASSIFY_FINE else 0
    
    # make the directory if it does not exist
    if not os.path.isdir(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)
    
    # load up all training data
    print('Processing training data')
    x_set, y_set = cifar100_binaries_to_tensor(SOURCE_PATH, TRAIN_FILENAME, N_TRAIN_IMAGES, n_labels, label_offset)
    
    TRAIN_FILENAME = os.path.join(OUTPUT_PATH, DatasetH5.train_file_prefix + '_00001' + DatasetH5.ext)
    write_h5_dataset(TRAIN_FILENAME, x_set, y_set) 
    
    # load up all test data
    print('Processing testing data')
    x_set, y_set = cifar100_binaries_to_tensor(SOURCE_PATH, TEST_FILENAME, N_TEST_IMAGES, n_labels, label_offset)
    
    TEST_FILENAME = os.path.join(OUTPUT_PATH, DatasetH5.test_file_prefix + '_00001' + DatasetH5.ext)
    write_h5_dataset(TEST_FILENAME, x_set, y_set)
    
    # Checking that it works
    print('Checking that dataset loads')
    cifar100 = DatasetH5(OUTPUT_PATH)
    
    print('Done')
