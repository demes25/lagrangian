# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Domains


from context import *
from geometry import Geometry
from composition import Function
import tensorflow as tf
from dataclasses import dataclass  # TODO: make IMAGE dataclass. include bookkeeping


# --- DOMAINS --- #

# this is a wrapped tensor object that discretizes a given range of values
# TODO: generalize to other discretization methods. right now it just makes a linear mesh
class Domain:
    # we discretize.
    #
    # the input is X [2, N] (the ranges) and dX [N] (the step sizes)
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
    def discretize(X : tf.Tensor, dX : tf.Tensor, indexing='ij') -> tf.Tensor: 

        grid_slices = []

        starts = X[0]
        ends = X[1]

        for i in range(tf.shape(starts)[0]):
            start = starts[i]
            end = ends[i]
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


    def __init__(
        self,
        geometry : Geometry, # a pair of functions, each of which return [B, N, N]
        ranges : tf.Tensor, # should be a [2, N] tensor,
        steps : tf.Tensor, # should be a [N] tensor  

        pad = 0, # we allow padding around the boundary for convolution purposes
        all_around = False # pad all_around, or pad only at the end of each axis.
    ):
        
        # this is N
        self.dimension = int(tf.shape(ranges)[1].numpy())
        

        # this is [n1, n2, ..., N] where n1, n2, ... are the amount of discrete points over coordinates x1, x2, ...
        if pad == 0:
            self.mesh = Domain.discretize(ranges, steps)
        else:
            padder = self.steps * pad
            if all_around: # pad evenly all around
                padding = tf.stack([-padder, padder], axis=0)
            else: # pad only at the end of each axis
                padding = tf.stack([tf.zeros_like(padder), padder], axis=0)

            self.mesh = Domain.discretize(self.ranges + padding, self.steps)

        self.pad = pad
        self.all_around = all_around

        # TODO: make a 'convolvable mesh' which has padded dimensions for convolvability purposes.
        # or do some other workaround for the padding problem

        self.ranges = ranges 
        self.steps = steps

        self.shape = tf.cast((ranges[1] - ranges[0])/steps, tf.int32)

        self._dtype = tf.as_dtype(self.mesh.dtype)

        # the functions act on flattened meshes 
        # (i.e. all coordinate mesh dimensions should be flattened to be the batch dimension)
        self.flattened = tf.reshape(self.mesh, shape=[-1, self.dimension])
        
        self.geometry = geometry
    

    # act some function on the mesh X : X[i1, i2, i3, ...] = (x1, x2, x3, ...)
    # - return a mesh F : F[i1, i2, i3, ...] = f(x1, x2, x3, ...)
    def image(self, f : Function, return_shape = False):
        f = f(self.flattened)

        shape = tf.shape(f)[1:]
        
        final_shape = tf.concat([self.mesh_shape, shape], axis=0)

        f = tf.reshape(f, final_shape)

        if return_shape:
            return f, shape.numpy().tolist() # returns the function output shape
        else:
            return f


