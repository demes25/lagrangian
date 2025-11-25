# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Lagrangians

from functionals import *
from algebra import Norm, dVol
from geometry import Minkowski

# Here, we will define Lagrangian generators
# These will be nested functionals that look like so:
#
# a Lagrangian generator K takes in some hyperparameters (*args), as well as a pair of metrics (g, g^-1)
# and returns a Lagrangian:
#       K : (*args, metrics) -> L
# 
# Wherefrom the Lagrangian is a functional which takes in a field/function A, and returns a function
# on batched points [B, ...] that gives the values of the lagrangians of the field A at the given points
#       L : A -> l
#       l : [B, ...] -> [B]


# --- REGULAR LAGRANGIAN GENERATOR --- #

# these will be functionals that take in hyperparameters
# and return functionals that take in a coordinate point
# and return the lagrangian at the given point value

def FreeParticle(
    m, # mass of the free particle
    metrics # geometry of our space
):  
    g, _ = metrics

    def _s(x):
        # X should be [B, N, 1] since X is directly position, 
        # and we want to find the kinetic term by mul.
        kinetic_term = Norm(diff(x), g)
        return scale_fn(half * m, kinetic_term)
    
    return _s 


# --- FIELD LAGRANGIAN DENSITY GENERATORS --- #

# these will be functionals that take in hyperparameters
# and return functionals that take in a field and return the lagrangian density at the given field value

# here we must multiply by volume form, since these are densities

# scalar field phi
def FreeScalarField(
    m = zero, # mass of the scalar field
    metrics = Minkowski(N=4) # metrics should be a tuple of tensor functions (g, g_inv)
):
    g, g_inv = metrics
    vol_form = dVol(g)

    # phi should be a functional [B, N] -> [B]
    if m == zero:
        def _s(phi):
            kinetic_term = Norm(grad(phi), g_inv)

            return mul_fn(vol_form, scale_fn(half, kinetic_term)) 

    else:
        def _s(phi):
            kinetic_term = Norm(grad(phi), g_inv) 
            mass_term = scale_fn(-m*m, mul_fn(phi, phi))

            return mul_fn(vol_form, scale_fn(half, add_fn(kinetic_term, mass_term))) 

    return _s


# free maxwell field A_mu
def FreeMaxwellField(
    metrics = Minkowski(N=4) # metrics should be a tuple of tensor functions (g, g_inv)
):
    g, g_inv = metrics
    vol_form = dVol(g)

    def _s(A):
        # A should be a functional [B, N] -> [B, N]

        f = grad(A) # [B, N, N] << last axis marks derivative variables
        g = apply_fn(tf.transpose, f, perm=(0, 2, 1))

        F = sub_fn(f, g) # F_mu nu = d_mu A_nu - d_nu A_mu

        kinetic_term = Norm(F, g_inv)

        return mul_fn(vol_form, scale_fn(-quarter, kinetic_term))

    return _s
        