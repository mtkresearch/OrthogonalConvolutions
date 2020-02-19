import os
import numpy as np
from common.datasets import onehot_encode

def cifar10_binaries_to_tensor(source_path, filenames):
    images_per_file = 10000
    n_x = 32
    n_y = 32
    n_channels = 3
    n_labels = 10
    bytes_per_image = n_channels * n_x * n_y
    bytes_per_label = 1

    datum_size = bytes_per_image + bytes_per_label
    file_size = images_per_file * datum_size
    
    data_buffer = np.zeros(len(filenames) * file_size, dtype='uint8')
    
    # load up all training data
    for index, filename in enumerate(filenames):
        with open(os.path.join(source_path, filename)) as data_file:
            data_buffer[index*file_size:(index+1)*file_size] = np.fromfile(data_file, dtype='uint8')
    
    y_set = onehot_encode(data_buffer[::datum_size], n_labels)
    image_data = np.delete(data_buffer, np.arange(0, data_buffer.size, datum_size))
    images = image_data.reshape(len(filenames) * images_per_file, n_channels, n_x, n_y).astype('float32') / 255
    x_set = np.transpose(images, (0, 2, 3, 1))

    return x_set, y_set
    
def cifar100_binaries_to_tensor(source_path, filename, n_images, n_labels, offset):
    n_x = 32
    n_y = 32
    n_channels = 3
    bytes_per_label = 2
    bytes_per_image = n_channels * n_x * n_y
    datum_size = bytes_per_image + bytes_per_label
    
    train_buffer = np.zeros(n_images * datum_size, dtype='uint8')
    
    # load up all training data
    with open(os.path.join(source_path, filename)) as data_file:
        data_buffer = np.fromfile(data_file, dtype='uint8')
    
    y_set = onehot_encode(data_buffer[offset::datum_size], n_labels)
    
    # remove both labels
    ranges = np.concatenate((np.arange(0, data_buffer.size, datum_size), np.arange(1, data_buffer.size, datum_size)))
    image_data = np.delete(data_buffer, ranges)
    
    images = image_data.reshape(n_images, n_channels, n_x, n_y).astype('float32') / 255
    x_set = np.transpose(images, (0, 2, 3, 1))
    

    return x_set, y_set
    
