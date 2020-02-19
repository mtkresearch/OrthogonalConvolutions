import logging 
import json

''' 
For performing checks on the config file.
'''
def check_section(config, version, section):
    if section not in config: raise KeyError('{section} section expected in config file'.format(section=section))
    if not isinstance(config[section], dict): raise TypeError('{section} should be a dictionary'.format(section=section))
    return config[section]

def check_list_section(config, version, section):
    if section not in config: raise KeyError('{section} section expected in config file'.format(section=section))
    if not isinstance(config[section], list): raise TypeError('{section} should be a list'.format(section=section))
    return config[section]

def check_experiment_args(config, version):
    experiment_args = check_section(config, version, 'experiment')
    if 'name' not in experiment_args: raise KeyError('experiment name required')
    if 'path' not in experiment_args: raise KeyError('experiment path required')
    if 'seed' not in experiment_args: raise KeyError('experiment seed required')
    return experiment_args

def check_dataset_args(config, version):
    dataset_args = check_section(config, version, 'dataset')
    if 'path' not in dataset_args: raise KeyError('dataset path required')
    if 'augment' in dataset_args and not isinstance(dataset_args['augment'], dict): raise TypeError('dataset augment parameter must be a dict')

    return dataset_args

def check_model_layers_args(model, version):
    layers_args = check_list_section(model, version, 'layers') 

def check_model_compile_args(model, version):
    compile_args = check_section(model, version, 'compile') 

def check_model_optimizer_args(model, version):
    optimizer_args = check_section(model, version, 'optimizer') 
    if 'name' not in optimizer_args: raise KeyError('optimizer name required')

def check_model_args(config, version):
    model_args = check_section(config, version, 'model')

    check_model_layers_args(model_args, version)
    check_model_compile_args(model_args, version)
    check_model_optimizer_args(model_args, version)

    return model_args 

def check_loss_args(config, version):
    if 'loss' in config: 
        return config['loss'] 
    else:
        return None

def check_training_args(config, version):
    training_args = check_section(config, version, 'training')
    if 'batch_size' not in training_args: raise KeyError('training batch_size required')
    if 'epochs' not in training_args: raise KeyError('training epochs required')

    return training_args 


def check_config(config):
    if config['version'] != '1.0': raise ValueError('Only supports version 1.0')
    version = config['version']

    experiment_args = check_experiment_args(config, version)
    dataset_args = check_dataset_args(config, version)
    model_args = check_model_args(config, version)
    loss_args = check_loss_args(config, version)
    training_args = check_training_args(config, version)

    logging.info(config)

    return config, experiment_args, dataset_args, model_args, loss_args, training_args

