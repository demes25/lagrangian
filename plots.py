# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Plotting

import tensorflow as tf
import tfplot

# Here I will write some functions using tfplot
# which is a tensorflow API for matplotlib
#
# we can wrap certain matplotlib functions to act directly on tensors

@tfplot.autowrap(figsize=(11, 11))
def heatmap(vals, fig=None, ax=None):

    im = ax.imshow(vals, extent=[-10.0, 10.0, -10.0, 10.0], cmap="viridis", origin="lower")

    # Optional colorbar
    fig.colorbar(im, ax=ax)
    return fig

def save_heatmap(mesh : tf.Tensor, name : str):
    pl = heatmap(mesh)
    tf.io.write_file(name, tf.io.encode_png(pl))



@tfplot.autowrap(figsize=(20, 11))
def plot(x, y, fig=None, ax=None):
    ax.plot(x, y)
    return fig

def save_plot(x : tf.Tensor, y : tf.Tensor, name : str):
    pl = plot(x, y)
    tf.io.write_file(name, tf.io.encode_png(pl))