import tensorflow as tf

# define global things here
DTYPE = tf.float32

# some useful c-numbers
zero = tf.constant(0, dtype=DTYPE)
one = tf.constant(1, dtype=DTYPE)
half = tf.constant(0.5, dtype=DTYPE)
quarter = tf.constant(0.25, dtype=DTYPE)

# PHYSICAL CONSTANTS
# I have set these to natural units
G = one 
c = one 
h = one
