# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Test

from observable import Observable
from domains import Domain
from operators import *
from keras.models import save_model, load_model
from keras.optimizers import AdamW
from geometry import Euclidean
from composition import *



# spring oscillation is repetitive in time - say we allow to train over 20 meters (should yield 20 oscillations)
step = 0.01
start = 1.0
end = 9.0 


# we'll take the value to be 0 at t=0 and x=0
# we want each point to look like [B, 1]

dX = tf.constant([step])
ranges = tf.constant([[start], [end]])
domain = Domain(Euclidean(1), ranges, dX) # this is now [b, b, 1]

squart, func_shape = domain.image(tf.sqrt, True)

squart_dif, squart_shape = Partials(domain)(squart, func_shape)

result = tf.stack([tf.squeeze(squart)[:-1], tf.squeeze(squart_dif)[:-1]], axis=1)

# i will make a plot here as well:

import tfplot

@tfplot.autowrap(figsize=(20, 11))
def plot(x, y, fig=None, ax=None):
    ax.plot(x, y)
    return fig
 
output = tf.squeeze(result)
domain = tf.squeeze(domain.mesh)[:-1]

pl = plot(domain, output)
tf.io.write_file("plot.png", tf.io.encode_png(pl))
