import os
import argparse
import tensorflow as tf
import json
import random
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from common.datasets import DatasetImagesH5
from representation.netx.layers import ModelFactory 
from representation.netx.optimizers import OptimizerFactory 
from representation.netx.config import * 
from imgaug import augmenters as iaa

def get_callbacks(path):
    """Creates a list of callbacks that can be used in the training procedure. 
    These include:
    - Tensorboard: logging the training metrics in real time to check progress.
    - ReduceLROnPlateau: reduces the learning rate when the validation accuracy does not
                         improve for 10 consecutive epochs.
    - ModelCheckpoint: saved the model that evaluates the best validaiton accuracy of the entire training
    Additional callbacks should be appended if wanted.

    Input: 
        path - string with th path to the folder where callbacks save data
    Output: 
        callbacks - a list of instances of the keras.callbacks.Callbacks class
    """
    # Set output folder for tensorboard and path for model with highest validation accuracy
    log_dir = os.path.join(path, 'tensorboard')
    model_file = os.path.join(path, 'model_val_accuracy.h5')
    # account for change in metric name across tensorflow versions
    learning_rate_metric = 'val_accuracy' if tf.__version__[0] == '2' else 'val_acc'

    # Make call backs list
    callbacks = []
    callbacks.append(TensorBoard(log_dir=log_dir))
    callbacks.append(ReduceLROnPlateau(monitor=learning_rate_metric))
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True, monitor=learning_rate_metric))
    return callbacks

if __name__ == '__main__':
    # Parse arguments from command line
    cli_parser = argparse.ArgumentParser(description='NetX classifier.')
    cli_parser.add_argument('--config', '-c', type=str, help='File that contains run configuration', default='config.json')
    args = cli_parser.parse_args()

    # Open config file 
    if args.config:
        with open(args.config) as json_file:
            config = json.load(json_file)
    else: 
        raise ValueError('No config file provided as input. Please provide using the --config flag.')

    # Check config dictionary and split in relevant sub-dictionaries 
    config_json, experiment_args, dataset_args, model_args, loss_args, training_args = check_config(config)

    # Check if path exists, otherwise create a folder
    if not os.path.isdir(experiment_args['path']): os.makedirs(experiment_args['path'])

    # Initiate random generator seeds
    np.random.seed(experiment_args['seed'])
    random.seed(experiment_args['seed'])
    tf.compat.v1.set_random_seed(experiment_args['seed'])

    # Load dataset and get generators, required for fit_generator and they process with data augmentation etc.
    dataset = DatasetImagesH5(**dataset_args)
    generator_train_data = dataset.generator_train_data
    generator_validation_data = dataset.generator_test_data
    
    # Make model
    model = ModelFactory(model_args['layers']).build(input_shape=dataset.input_shape)

    # Make optimizer and compile model with the optimizer
    optimizer = OptimizerFactory().build(**model_args['optimizer'])
    model.compile(**model_args['loss'], **model_args['compile'], optimizer=optimizer)

    # Batch size needs to be set manually for fit_generators and removed
    batch_size = training_args.pop('batch_size') 
    generator_train_data.batch_size = batch_size
    generator_validation_data.batch_size = batch_size

    # Training
    history = model.fit_generator(generator=generator_train_data, validation_data=generator_validation_data, callbacks=get_callbacks(experiment_args['path']), **training_args)

    # Save model at the end of training 
    model.save(os.path.join(path, 'model.h5'))