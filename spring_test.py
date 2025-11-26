# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Test

from observable import Observable
from system import System, discretize
from variations import SpringParticle
from keras.models import save_model, load_model
from keras.optimizers import AdamW
from geometry import Euclidean
from functionals import *

# Let's test this out by looking at a wave equation with fixed 
# starting points.

# we let our wave amplitude be an observable 
phi = Observable(activation='gelu')

# Now we create our spring system
SpringSystem = System(
    parameter_num=1, # two parameters - t, x

    geometry=Euclidean(N=1),

    boundary_function = constant_fn(zero), # we want the boundary to be fixed - say at 0. 
    variation_generator = SpringParticle, # we look at a free scalar field
    
    # variation hyperparameters:
    m=one, # we let unit mass
    k=tf.constant(2.0 * 3.1415) # we plan for one oscillation every unit of distance
)

# spring oscillation is repetitive in time - say we allow to train over 20 meters (should yield 20 oscillations)
step = 0.5
start = 0.0
end = 20.0 


# we'll take the value to be 0 at t=0 and x=0
# we want each point to look like [B, 1]

dX = tf.constant([step])
ranges = tf.constant([[start], [end]])
domain = discretize(ranges, dX) # this is now [b, b, 1]

left_boundary = domain[0] 
right_boundary = domain[-1] 

boundary = ranges
domain = tf.reshape(domain, [-1, 1]) # we reshape to [B, 1] = [bb, 1]


SpringSystem.train(phi, domain, boundary, AdamW(learning_rate=0.0001), epochs=200, boundary_weight=0.99)


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
