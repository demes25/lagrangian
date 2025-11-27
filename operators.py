# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operators


from context import *
from domains import Domain
import tensorflow as tf

from typing import Sequence, Callable, Tuple, Any


# TODO: FINISH THIS, DEBUG.

# we define kernels for differential operators
# these will act on an N-dimensional mesh
#
# our meshes will look like [n1, n2, ..., nN, N]
# so our function outputs will look like [n1, n2, ..., nN, *dimF]
#
# this gives back the kernel corresponding to the i'th partial derivative
# when applied to our function outputs.

# we follow standard procedure, right-handed: 
#           df = f(x+dx) - f(x)
# though could also left-handed:     
#           df = f(x) - f(x-dx)
# or centered:
#           df = [f(x+dx) - f(x-dx)]/2
partial_base = tf.constant([-1, 1], dtype=DTYPE)

def Kernel(base, dims, func_size, i) -> tf.Tensor:
    # this gives a differential kernel along the given axis
    # without dividing by the relevant dx.
    # so this takes in F and returns dF

    sizes = [1] * dims
    sizes[i] = 2
    
    _base = tf.reshape(base, sizes + [1, 1])

    # reshape to [k1, k2, ..., kN, func_size, func_size]
    return tf.broadcast_to(_base, sizes + [func_size, func_size])


# --- some help --- #

# we follow standard boilerplate for acting the above differntial operators
# on some mesh ('substrate') 
def prep_for_kernel(substrate : tf.Tensor, mesh_shape, func_shape):
    # we expect an input channel dim and an output channel dim for convolution.
    # in this case we reshape so that the input channel dim is always 1.
    # but the output channel dim might change depending on the output shape of our function.
    
    # mesh_shape is the shape of the mesh - i.e. the whole [n1, n2, ..., N] leading up to everything
    # and func_shape is the output shape of the function.

    # in all, substrate shape should look like   mesh_shape + func_shape

    func_size = tf.reduce_prod(func_shape)

    if tf.size(func_shape) == 0:
        # expand out so our function out size becomes 1
        substrate = tf.expand_dims(substrate, axis=-1)
        # this will just be our default treatment of scalars - expand them to have dimension 1.
    
    else:
        # we compress our function output shape - we will later reshape this back
        reduction_shape = tf.concat([mesh_shape, [func_size]], axis=0)
        substrate = tf.reshape(substrate, shape=reduction_shape)
    
    substrate = tf.expand_dims(substrate, axis=0)

    # we return the prepped substrate,
    # and the relevant dim_out for convolution also
    return (substrate, func_size)


# now we define operators.
#
# these should take in (function_mesh, function_output_shape)
# and output (result_mesh, result_output_shape)

_Shape = Sequence | tf.TensorShape | Any
Operator = Callable[[tf.Tensor, _Shape], Tuple[tf.Tensor, _Shape]]

def Partials(domain : Domain, axes = []) -> Operator:
    # axes should be a list of axes over which to compute partials

    # returns the list of relevant partial derivatives, same order as in axes,
    # of f over the given domain.
    def _h(substrate : tf.Tensor, func_shape : _Shape) -> Tuple[tf.Tensor, _Shape]:
        substrate_shape = tf.concat([domain.shape, func_shape], axis=0)

        substrate, func_size = prep_for_kernel(substrate, domain.shape, func_shape)

        outputs = []

        # if axes are not specified, we calculate over all variables by default.
        for i in (axes if axes else range(domain.dimension)):
            kernel = Kernel(partial_base, domain.dimension, func_size, i)
            post_kernel = tf.nn.convolution(substrate, kernel)
            output = tf.reshape(post_kernel, substrate_shape)

            outputs.append(output/domain.steps[i]) # divide by relevant dx
        
        # stacks along the final axis
        return tf.stack(outputs, axis=-1), func_shape + [len(axes) if axes else domain.dimension]
    
    return _h 





