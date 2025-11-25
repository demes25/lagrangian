from geometry import Schwarzschild, tf

g, g_inv = Minkowski(tf.constant(10, dtype=tf.float32))

print(g(tf.constant([[0, 1, 1, 0]], dtype=tf.float32)))
print(g_inv(tf.constant([[0, 1, 1, 0]], dtype=tf.float32)))