from geometry import Schwarzschild, tf, Minkowski
from algebra import dVol

g, g_inv = Schwarzschild(tf.constant(10, dtype=tf.float32))
P = tf.constant([[0, 1, 1, 0]], dtype=tf.float32)

print(g(P))
print(g_inv(P))
print(dVol(g)(P))