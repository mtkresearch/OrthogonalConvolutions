import tensorflow
from tensorflow.keras.layers import Layer, Conv2D, Permute, Activation, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class NetXCycle(Layer):
    """
    NetXCycle layer is a block built of three convolutions along different axis. 
    Each convolution is followed by other standard layers, such as batch norm and dropout, depending on the defined parameters.
    """
    def __init__(self, output_shape, kernel_shape=(2,2), activation='relu', dropout_rate=0.0, kernel_regularizer=None, **kwargs):
        '''
        Input:
            output_shape (required) - list or tuple of 3 integers defining the output shape of the layer (excluding the batch_size dimension)
            kernel_shape - tuple of 2 integers defining the kernel size of the convolutions
            activation - activation function, such as tf.nn.relu, or string name of built-in activation function, such as "relu"
            dropout_rate - float between 0 and 1
            kernel_regularizer -  tf.keras.initializers.Initializer for the kernel weights matrix 
        '''
        assert(len(output_shape) == 3)
        assert(len(kernel_shape) == 2)
        super(NetXCycle, self).__init__(**kwargs)

        self._output_shape = output_shape
        self._kernel_shape = kernel_shape
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._kernel_regularizer = kernel_regularizer

        # input_shape has (batch_size, x, y, z[features])
        # permutation: x y z -> z' x y -> y' z' x -> x' y' z'
        self._conv2D_a = Conv2D(filters=self._output_shape[2],kernel_size=self._kernel_shape,kernel_regularizer=self._kernel_regularizer,padding='same')
        self._conv2D_b = Conv2D(filters=self._output_shape[1],kernel_size=self._kernel_shape,kernel_regularizer=self._kernel_regularizer,padding='same')
        self._conv2D_c = Conv2D(filters=self._output_shape[0],kernel_size=self._kernel_shape,kernel_regularizer=self._kernel_regularizer,padding='same')
        
        self._batch_norm_a = BatchNormalization(axis=3)
        self._batch_norm_b = BatchNormalization(axis=1)
        self._batch_norm_c = BatchNormalization(axis=2)

        # need separate convolutions - we expect weights to differ
        self._dropout_a = Dropout(rate=self._dropout_rate)
        self._dropout_b = Dropout(rate=self._dropout_rate)
        self._dropout_c = Dropout(rate=self._dropout_rate)

        self._permute = Permute((3,1,2))
        self._activation_layer = Activation(self._activation) 


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert(len(input_shape) == 4)
        super(NetXCycle, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert(len(x.shape) == 4)
        
        x = self._conv2D_a(x)
        x = self._batch_norm_a(x)
        x = self._activation_layer(x)
        x = self._dropout_a(x)
        x = self._permute(x)

        x = self._conv2D_b(x)
        x = self._batch_norm_b(x)
        x = self._activation_layer(x)
        x = self._dropout_b(x)
        x = self._permute(x)

        x = self._conv2D_c(x)
        x = self._batch_norm_c(x)
        x = self._activation_layer(x)
        x = self._dropout_c(x)
        x = self._permute(x)

        return x 

    def get_config(self):
        # need to override this
        config = super(NetXCycle, self).get_config().copy()
        config.update({
           'output_shape': self._output_shape,
           'kernel_shape': self._kernel_shape,
           'activation': self._activation,
           'dropout_rate': self._dropout_rate,
        })
        return config
