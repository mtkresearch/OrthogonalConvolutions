import os
import numpy as np
from cifar_helpers import cifar10_binaries_to_tensor
from common.datasets import DatasetH5, write_h5_dataset
import PIL
from PIL import Image

# We process binary into a h5 format compatible with DatasetH5, so that it the same architecture can be used over different datasets
# both on the farm and on our workstations
#PATH = '/data/raw/pathfinder/curv_contour_length_14/'
PATH = '/data/raw/pathfinder/curv_baseline/'
METADATA_PATH = os.path.join(PATH, 'metadata/1.npy')
OUTPUT_PATH = '/data/pathfinder/length_6'
    
if __name__ == '__main__':
    # make the directory if it does not exist
    if not os.path.isdir(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)
    
    metadata = np.load(METADATA_PATH)

    train_batch_size = 40000
    test_batch_size = 10000
    x_size = 128
    y_size = 128
    batch_path_index = 0
    filename_index = 1
    linked_index = 3

    x_set = np.zeros([train_batch_size, x_size, y_size, 1])
    y_set = np.zeros([train_batch_size,1])

    print('Processing training data')

    for index in range(train_batch_size):
        batch_path  = metadata[index][batch_path_index]
        filename = metadata[index][filename_index]
        img = Image.open(os.path.join(PATH, batch_path, filename))
        x_set[index,:,:,0] = np.asarray(img.resize((x_size, y_size))) / 255.0
        y_set[index,0] = metadata[index][linked_index]

    train_filename = os.path.join(OUTPUT_PATH, DatasetH5.train_file_prefix + '_00001' + DatasetH5.ext)
    write_h5_dataset(train_filename, x_set, y_set)

    x_set = np.zeros([test_batch_size, x_size, y_size, 1])
    y_set = np.zeros([test_batch_size, 1])

    print('Processing testing data')

    for test_index in range(test_batch_size):
        index = train_batch_size + test_index
        batch_path  = metadata[index][batch_path_index]
        filename = metadata[index][filename_index]
        img = Image.open(os.path.join(PATH, batch_path, filename))
        x_set[test_index,:,:,0] = np.asarray(img.resize((x_size, y_size))) / 255.0
        y_set[test_index,0] = metadata[index][linked_index]

    test_filename = os.path.join(OUTPUT_PATH, DatasetH5.test_file_prefix + '_00001' + DatasetH5.ext)
    write_h5_dataset(test_filename, x_set, y_set)

    # Checking that it works
    print('Checking that dataset loads')
    pathfinder = DatasetH5(OUTPUT_PATH)

    train_gen = pathfinder.generator_train_data

    for index in range(train_batch_size):
        batch_path  = metadata[index][batch_path_index]
        filename = metadata[index][filename_index]
        img = Image.open(os.path.join(PATH, batch_path, filename))
        print(train_gen[0][index][:,:,0] == np.asarray(img.resize((x_size, y_size))) / 255.0)

    print('done')
