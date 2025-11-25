# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Systems

import tensorflow as tf
from keras.optimizers import Optimizer
from keras.models import Model
from keras.utils import Progbar

from functionals import *
from algebra import Norm
from typing import Callable

# here we look at general systems and how to deal with them:
# discretizing the domain, passing through neural networks, 
# applying lagrangians and boundary conditions

# we discretize.
#
# the input is X [N, 2] (the ranges) and dX [N] (the step sizes)
#
# where N is the dimension count, the first col tells us starting points and 
# second col tells us ending points.
#
# dX tells us how far away points should be in each dimension.
#
# so this will give an [n**N, N] grid
#
# for now this splits our ranges linearly via tf.linspace
# but this is slightly suspect for more nontrivial geometries, like spherical
# still, though, should work not too bad.
#
# we use meshgrid to produce a cartesian product of linspaces
# as such I will make the indexing adjustable depending on how we want it
def discretize(X, dX, indexing='xy'): 

    grid_slices = []

    for i in range(tf.shape(X)[0]):
        start = X[i, 0]
        end = X[i, 1]
        spacing = dX[i]
        
        # TODO: this can be made more general to take 'spacing' as literal distance
        # and use the metric at a given point to find the next discrete point by this measure of distance
        
        n = (end - start)/spacing # we find the amount of points
        n = tf.cast(n, tf.int32) # must be integer

        new_slice = tf.linspace(start, end, n) # from x[0] to x[1], n discrete points

        grid_slices.append(new_slice)
    
    grid = tf.meshgrid(*grid_slices, indexing=indexing)

    # then we stack along the last axis
    return tf.stack(grid, axis=-1)



class System:
    def __init__(
        self,
        parameter_num, # how many parameters do we have? (4 for fields, 1 (time) for regular)

        metrics,  # (g, g^-1)
        # the metrics for our underlying geometry. we will use this to generate the lagrangian
        # and calculate the boundary penalty
        # defaults to euclidean if None
    
        boundary_function, # function that gives us the desired boundary values

        lagrangian_generator, 
        # a lagrangian generator from the 'lagrangians' module.
        # note that this thing is a function that takes in a field/soltuion A
        # and returns a function L[A] which acts on coordinates

        *lagrangian_args, # args for the lagrangian generator
        **lagrangian_kwargs, # kwargs for the lagrangian generator
    ):
        self.parameter_num = parameter_num
        self.lagrangian = lagrangian_generator(*lagrangian_args, **lagrangian_kwargs, metrics=metrics)
        self.boundary_function = boundary_function
        self.metric, self.inv_metric = metrics

    # I will assume that all proposed solutions are either scalar
    # or covariant, and therefore I will be taking norms using the inverse metric 
    # (as opposed to the regular metric, as one would for contravariant objects)


    # we calculate the action for some 
    # proposed solution U
    #
    # this returns a function S : [B, N] -> []
    # which takes in a discretized domain and returns the action of U over that domain
    #
    # note that, since all of our lagrangian generators already include the volume form,
    # we are free to simply sum without worrying about weighting.
    def action(self, U):
        return apply_fn(tf.reduce_mean, self.lagrangian(U), axis=0)
    

    # we calculate the boundary penalty for some
    # proposed solution U
    #
    # this returns a function P : [B, N] -> []
    # which takes in a discretized boundary and returns the mean square difference between U and our
    # boundary function over the given points.
    def boundary_penalty(self, U):
        # we first take differences
        dif = sub_fn(U, self.boundary_function)
        
        # then we take norms (Norm remains squared, recall)
        per_point_penalty = Norm(dif, self.inv_metric) 

        # and average over all points
        return apply_fn(tf.reduce_mean, per_point_penalty, axis=0)


    # this returns a loss functor for the proposed field U
    # the idea is, we encapsulate everything like so:
    # 
    # this functor returns:
    #   a function which takes in a domain, a boundary, and a
    #   BOUNDARY WEIGHT (how much to focus on the boundary loss)
    #   and returns the relevant convex combination of action and boundary loss 
    def loss_functor(self, U):
        action = self.action(U)
        boundary_penalty = self.boundary_penalty(U)

        def loss(domain, boundary, boundary_weight):
            # assert 0 <= boundary_weight <= 1

            if boundary_weight == zero:
                return action(domain)
            elif boundary_weight == one:
                return boundary_penalty(boundary)
            else:
                return (1-boundary_weight) * action(domain) + boundary_weight * boundary_penalty(boundary)

        return loss


    # we allow an arbitrary boundary weight function which returns some weight depending on which
    # epoch we are on. this allows us to focus on the boundary at different points through training
    def train(
        self, 
        U : Model, 
        domain, # domain over which to train
        boundary, # boundary over which to train
        optimizer : Optimizer, 
        epochs = 10,
        boundary_weight : Callable[[int], float] | float = 0.5 # boundary weight function
    ):
        # we put a nice little progress bar for prettiness
        bar = Progbar(epochs)

        loss = self.loss_functor(U)
        
        # if boundary_weight is a constant - just go about as normal
        if not callable(boundary_weight): 
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    loss_value = loss(domain, boundary, boundary_weight)

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

                bar.update(epoch + 1, values=[('loss', loss_value.numpy())]) 

        # otherwise we evaluate it per-epoch.
        else:
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    loss_value = loss(domain, boundary, boundary_weight(epoch)) 

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

                bar.update(epoch + 1, values=[('loss', loss_value.numpy())]) 

