# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Spacetimes

from functionals import *
from algebra import Delta, Eta

# define some spacetimes

# dimensionality
S = 3 # spatial dim
N = S + 1 # spatiotemporal dim

# each batched spatial tensor will be assumed to look like [B, S, ...]
# each batched spacetime tensor will be assumed to look like [B, N, ...]

# metrics should be tensor-to-tensor functions,
# which take in a batched rank 1 tensors (coordinates) and return batched rank-2 tensors

# spacetimes (or spaces) should be functionals which take in some parameter (if needed)
# and return a tuple -- (metric, inverse metric) -- of metric functions
# (note that this does not coincide exactly with what a spacetime actually is - our functionals will also assume a fixed coordinate map)

# --- SPACES --- # 

# takes in None
# returns 2-tuple of constant functions fn : [B, S] 
def EuclideanSpace():
    # x must be [B, S]
    def _m(x):
        return Delta(S)
    
def SphericalSpace():
    # X must be [B, S]
    def _metric(X):
        # [r, theta, phi]
        R = X[:, 0]
        Theta = X[:, 1]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([one, rsq, rsq * sinth * sinth], axis=-1)

        return tf.linalg.diag(diag_elements)

    
    # X must be [B, N]
    def _inv_metric(X):
        # [r, theta, phi]
        R = X[:, 0]
        Theta = X[:, 1]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([one, one/rsq, one/(rsq * sinth * sinth)], axis=-1)

        return tf.linalg.diag(diag_elements)
        
    return (_metric, _inv_metric)

# --- SPACETIMES --- #

# takes in None
# returns 2-tuple of constant functions fn : [B, N] -> B ** eta
def MinkowskiSpacetime():
    # X must be [B, N]
    def _m(X):
        B = tf.shape(X)[0]
        return tf.tile(Eta, (B, N, N))
    
    return (_m, _m)

# takes in scalar
# returns a tuple of fn : [B, N] -> [B, N, N]
# (g, g^-1)
def SchwarzschildSpacetime(
    M # black hole mass
):
    schw_radius = tf.constant(2 * M, dtype=DTYPE) * G/(c*c)

    # X must be [B, N]
    def _metric(X):
        # [t, r, theta, phi]
        R = X[:, 1]
        Theta = X[:, 2]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4
        U = one - schw_radius/R #  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([-U, one/U, rsq, rsq * sinth * sinth], axis=-1)

        return tf.linalg.diag(diag_elements)

    
    # X must be [B, N]
    def _inv_metric(X):
        # [t, r, theta, phi]
        R = X[:, 1]
        Theta = X[:, 2]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4
        U = one - schw_radius/R #  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([-one/U, U, one/rsq, one/(rsq * sinth * sinth)], axis=-1)

        return tf.linalg.diag(diag_elements)
        
    return (_metric, _inv_metric)

