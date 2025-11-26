# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operators

import tensorflow as tf
from context import *
from typing import Callable

# TODO: wrap Functions in a bona-fide class so we can know things like input/output dimensions, etc.

# here we define algebra for functional addition, multiplication, more general operators
# also we define gradients via tf.GradientTape
Function = Callable[[tf.Tensor], tf.Tensor]

# we define the constant function - if necessary
def constant_fn(c : tf.Tensor) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        return c
    return _h

# for all these, *fnargs is assumed to be a bunch of functions/functionals

def add_fn(*fnargs : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        return tf.add_n([f(x) for f in fnargs])
    return _h
    
def mul_fn(*fnargs : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        # no such 'mul_n' so we multiply manually
        result = fnargs[0](x)
        for i in range(1, len(fnargs)):
            result = result * fnargs[i](x)
        return result
    return _h

def apply_fn(fn : Callable, *fnargs : Function, **kwargs):
    # we assume fnargs contains ONLY FUNCTIONS
    # and we apply 'fn' to them at some point x.
    # any other arguments to fn may be included in kwargs
    def _h(x : tf.Tensor) -> tf.Tensor:
        eval_fnargs = [f(x) for f in fnargs]
        return fn(*eval_fnargs, **kwargs)

    return _h 


def scale_fn(c : tf.Tensor, f : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        return c * f(x)
    return _h

def sub_fn(f : Function, g : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        return f(x) - g(x)
    return _h 

def div_fn(f : Function, g : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        return f(x)/g(x)
    return _h


# we define some operators as well

# F a rank n tensor function of a scalar input
# returns F' a rank n tensor (single-variable differentiation)
def diff(F : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # we watch our inputs
            tape.watch(x)

            # returns rank n+1 tensor F', where F'[..., mu] = partial_mu F  
            f = F(x)
        return tape.gradient(f, x)
    return _h

# F a rank n tensor function
# returns F' a rank n+1 tensor (multi-variable differentiation)
def grad(F : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(persistent=True) as tape:
            # we watch our inputs
            tape.watch(x)

            # returns rank n+1 tensor F', where F'[..., mu] = partial_mu F  
            f = F(x)

        if tf.rank(f) > 1:
            return tape.batch_jacobian(f, x)
        
        return tape.gradient(f, x)
    
    return _h


# G a rank 2 tensor function
def det(G : Function) -> Function:
    def _h(x : tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(G(x))
    
    return _h