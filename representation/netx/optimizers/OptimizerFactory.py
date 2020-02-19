import tensorflow
import logging
from tensorflow.keras import optimizers 
from tensorflow.keras import Model

class OptimizerFactory:
    optimizer_dict = {
        'Adadelta': optimizers.Adadelta,
        'Adagrad': optimizers.Adagrad,
        'Adam': optimizers.Adam,
        'Adamax': optimizers.Adamax,
        'Nadam': optimizers.Nadam,
        'RMSprop': optimizers.RMSprop,
        'SGD': optimizers.SGD
    }

    def build(self, name, args=None):
        '''
        {
            "name": "name"
            "args: {"args of optimizer"}
        },... 
        '''

        return OptimizerFactory.optimizer_dict[name]() if not args else OptimizerFactory.optimizer_dict[name](**args)
