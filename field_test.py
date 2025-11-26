# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Test

from observable import Observable
from system import System, discretize
from variations import FreeScalarField
from keras.models import save_model, load_model
from keras.optimizers import Adam
from geometry import Minkowski
from functionals import *

# Let's test this out by looking at a wave equation with fixed 
# starting points.

# we let our wave amplitude be an observable 
phi = Observable(activation='gelu')

# spring oscillation is repetitive in time - say we allow to train over 20 meters
step = 0.01
start = 0.0
end = 5.0


# we'll take the value to be 0 at t=0 and x=0
# we want each point to look like [B, 2]

dX = tf.constant([step, step])
ranges = tf.constant([[start, start], [end, end]])
domain = discretize(ranges, dX) # this is now [b, b, 2]

left_boundary = domain[:, 0] - step # x ~= 0, we offset slightly
right_boundary = domain[:, -1] + step # x ~= 100

boundary = tf.concat([left_boundary, right_boundary], axis=0)
domain = tf.reshape(domain, [-1, 2]) # we reshape to [B, 2] = [bb, 2]

waveSystem = System(
    parameter_num = 2, # 2 variables - t and x
    geometry = Minkowski(N=2), # minkowski 2-space
    boundary_function=constant_fn(zero), # zero on the boundary
    variation_generator=FreeScalarField,
    
    m=one # we allow mass-one
)



waveSystem.train(phi, domain, boundary, Adam(), epochs=100, boundary_weight=0.25)


save_model(phi, 'spring.keras')

# i will make a plot here as well:

import tfplot

@tfplot.autowrap(figsize=(20, 11))
def plot(x, y, fig=None, ax=None):
    ax.plot(x, y)
    return fig
 
output = tf.squeeze(phi(domain))
domain = tf.squeeze(domain)

spring_plot = plot(domain, output)
tf.io.write_file("spring_plot.png", tf.io.encode_png(spring_plot))
