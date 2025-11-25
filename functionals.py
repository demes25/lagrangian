# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operators

import tensorflow as tf
from context import *

# here we define algebra for functional addition, multiplication, more general operators
# also we define gradients via tf.GradientTape


# we define the constant function - if necessary
def constant_fn(c):
    def _h(x):
        return c
    return _h

# for all these, *fnargs is assumed to be a bunch of functions/functionals

def add_fn(*fnargs):
    def _h(x):
        return tf.add_n([f(x) for f in fnargs])
    return _h
    
def mul_fn(*fnargs):
    def _h(x):
        # no such 'mul_n' so we multiply manually
        result = fnargs[0](x)
        for i in range(1, len(fnargs)):
            result = result * fnargs[i](x)
        return result
    return _h

def apply_fn(fn, *fnargs, **kwargs):
    # we assume fnargs contains ONLY FUNCTIONS
    # and we apply 'fn' to them at some point x.
    # any other arguments to fn may be included in kwargs
    def _h(x):
        eval_fnargs = [f(x) for f in fnargs]
        return fn(*eval_fnargs, **kwargs)

    return _h 


def scale_fn(c, f):
    def _h(x):
        return c * f(x)
    return _h

def sub_fn(f, g):
    def _h(x):
        return f(x) - g(x)
    return _h 

def div_fn(f, g):
    def _h(x):
        return f(x)/g(x)
    return _h


# we define some operators as well

# F a rank n tensor function
# returns F' a rank n+1 tensor
def grad(F):
    def dF(x):
        with tf.GradientTape(persistent=True) as tape:
            # we watch our inputs
            tape.watch(x)

            # returns rank n+1 tensor F', where F'[..., mu] = partial_mu F  
            f = F(x)
        return tape.batch_jacobian(f, x)
    
    return dF

# G a rank 2 tensor function
#
def det(G):
    def _d(x):
        return tf.linalg.det(G(x))
    
    return _d