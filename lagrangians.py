# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Lagrangians

from functionals import *
from algebra import Norm, dVol
from geometry import MikowskiSpacetime


# --- REGULAR LAGRANGIANS --- #

# these will be functionals that take in hyperparameters
# and return functionals that take in a coordinate point
# and return the lagrangian at the given point value

def FreeParticle(
    m = one
    #metrics = TODO: generalize this to any spatial metric
):
    def _s(x):
        kinetic_term = Norm(grad(x))
        return scale_fn(half, kinetic_term)
    
    return _s 


# --- FIELD LAGRANGIAN DENSITIES --- #

# these will be functionals that take in hyperparameters
# and return functionals that take in a field and return the lagrangian density at the given field value

# here we must multiply by volume form, since these are densities

# scalar field phi
def FreeScalarField(
    m = zero, # mass of the scalar field
    metrics = MikowskiSpacetime() # metrics should be a tuple of tensor functions (g, g_inv)
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
    metrics = MikowskiSpacetime() # metrics should be a tuple of tensor functions (g, g_inv)
):
    g, g_inv = metrics
    vol_form = dVol(g)

    def _s(A):
        # A should be a functional [B, N] -> [B, N]

        f = grad(A) # [B, N, N] << last axis marks derivative variables
        g = apply_fn(tf.transpose, f, perm=(0, 2, 1))

        F = sub_fn(f, g) # F_mu nu = d_mu A_nu - d_nu A_mu

        kinetic_term = Norm(F, g_inv)

        return mul_fn(vol_form, scale_fn(half, kinetic_term))

    return _s
        