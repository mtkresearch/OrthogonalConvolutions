import tensorflow
import logging
from .OrthogonalConv import OrthogonalConv
from .BilinearInterpolate3D import BilinearInterpolate3D
from tensorflow.keras import layers, Model, regularizers

class ModelFactory:
    layer_dict = {
        'OrthogonalConv': OrthogonalConv, 
        'Conv2D': layers.Conv2D,
        'Dense': layers.Dense,
        'Conv3D': layers.Conv3D,
        'Reshape': layers.Reshape,
        'Dropout': layers.Dropout,
        'MaxPooling2D': layers.MaxPooling2D,
        'SpatialDropout2D': layers.SpatialDropout2D,
        'BatchNormalization': layers.BatchNormalization,
        'Flatten': layers.Flatten,
        'Conv1D': layers.Conv1D,
        'UpSampling2D': layers.UpSampling2D,
        'UpSampling3D': layers.UpSampling3D,
        'MaxPooling2D': layers.MaxPooling2D,
        'MaxPooling3D': layers.MaxPooling3D,
        'AveragePooling2D': layers.AveragePooling2D,
        'AveragePooling3D': layers.AveragePooling3D,
        'Activation': layers.Activation,
        'BilinearInterpolate3D': BilinearInterpolate3D
    }

    def __init__(self, layers):
        '''
        layers has the format:
        [{
            "name": "layer name form layer_dict",
            "args: "args of layer"
         },... 
        ]
        '''
        self._layers = layers

    def build(self, input_shape=None, name=None):
        '''
        Create a model 
        '''
        if input_shape:
            inputs = layers.Input(input_shape)
        else: 
            raise ValueError('Either input_shape or an input tensor must be provided')
        
        network = inputs
        for index, layer in enumerate(self._layers):
            logging.info("Building layer " + str(index + 1) + " -- " + layer['name'])
            network = ModelFactory.add_layer(network, layer)

        return Model(inputs = inputs, outputs = network) 
    
    @staticmethod
    def make_regularizer(name, args):
        if name == 'l1':
            return regularizers.l1(args['rate'])
        elif name == 'l2':
            return regularizers.l2(args['rate'])
        else:
            raise ValueError('Unsupported regularizer: ' + name + ' requested')

    @staticmethod
    def add_layer(inputs, layers):
        '''
        adds a single layer or a list of layers. 
        params: layer: a dictionary of the form name: (name), args: (arguments for layer constructor) 
        or a list of such.
        '''
        if not isinstance(layers, list):
            layers = [layers]

        network = inputs
        for layer in layers:
            if 'args' in layer:

                if 'kernel_regularizer' in layer['args']:
                    params = layer['args'].pop('kernel_regularizer')
                    layer['args']['kernel_regularizer'] = ModelFactory.make_regularizer(name=params['name'], args=params['args'])

                if isinstance(layer['args'], dict):
                    network = ModelFactory.layer_dict[layer['name']](**layer['args'])(network)
                elif isinstance(layer['args'], list):
                    network = ModelFactory.layer_dict[layer['name']](*layer['args'])(network)
                else:
                    raise TypeError('Layer args should be a list or a dictionary')
            else:
                network = ModelFactory.layer_dict[layer['name']]()(network)
        return network
