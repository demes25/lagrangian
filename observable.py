# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Fields

# this is a quick demonstration of a neural network that
# offers some learned solution to a given lagrangian density over 
# a given domain.

import tensorflow as tf
from keras import models, layers, utils
from context import DTYPE, PACKAGE

# wraps a simple Sequential model
# 
# we apply a neural network to some discretization of our desired domain, essentially
# finite element method, pass everything through a dense, and train on minimizing the total action.
#
# these will be either spacetime fields - each point in the domain has four coordinates.
# or spatial observables - like position - each point in the domain has one coordinate.
#
# Note that this is a Callable that takes in a tf.Tensor and returns a tf.Tensor,
# so it can be used anywhere that a 'Function' type is mentioned as defined in the 'functionals' module.
@utils.register_keras_serializable(package=PACKAGE)
class Observable(models.Model):
    
    def __init__(self, 
                 shape = [], # by default a scalar field
                 hidden_dims = 64, # the hidden dimension of our mlp layers 
                 activation = 'gelu', # activation - i will put gelu for balance between universal differentiability and neuron expressivity - sigmoids suppress large numbers
                 dtype = DTYPE, # we choose dtype, if wanted
                 **kwargs
                 ):
        
        dtype = tf.as_dtype(dtype)

        super().__init__(dtype=dtype, **kwargs)

        # for serializability
        self.internal_config = {
            'shape' : shape,
            'hidden_dims' : hidden_dims,
            'activation' : activation,
            'dtype' : dtype.name
        }

        self.shape = shape
        self._dtype = dtype
        
        if len(shape) == 0:
            flat_shape = 1
        else:
            flat_shape = sum(shape) # output shape flattened for Dense calculation

        # we will make a three-layer mlp
        self.fxn = models.Sequential(
            layers= [
                layers.Dense(hidden_dims, activation=activation, dtype=dtype), 
                layers.Dense(hidden_dims, activation=activation, dtype=dtype),
                layers.Dense(hidden_dims, activation=activation, dtype=dtype), # note that gelu suppresses negative numbers
                layers.Dense(flat_shape, dtype=dtype) # so we put no activation for last layer, we want to be able to have negative numbers
            ]
        )

    # we add the standard boilerplate for serializability
    @classmethod
    def from_config(cls, config : dict) -> 'Observable':
        return cls(**config)

    def get_config(self) -> dict:
        return {**self.internal_config, **super().get_config()}
    
    # again boilerplate for building
    def build(self, input_shape):
        dummy_input = tf.zeros(shape=input_shape, dtype=self._dtype)
        _ = self.fxn(dummy_input)
        super().build(input_shape)


    # input : domain [B, n] - a set of B discrete n-coordinate events in spacetime
    # output : [B, S] - a set of B tensors corresponding to the field values at each given point 
    def call(self, domain : tf.Tensor) -> tf.Tensor:
        B = tf.shape(domain)[0]
        B = tf.expand_dims(B, axis=0)

        result = self.fxn(domain)
        result = tf.reshape(result, tf.concat([B, self.shape], axis=0))
        return result 
        
    
