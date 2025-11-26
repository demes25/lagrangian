# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Lagrangians

from functionals import *
from algebra import Norm, dVol
from geometry import Geometry, Minkowski

# Here, we will define Lagrangian generators
# These will be nested functionals that look like so:
#
# a Lagrangian generator K takes in some hyperparameters (*args), as well as a pair of metrics - i.e. a "geometry" (g, g^-1)
# and returns a Lagrangian:
#       K : (*args, geometry) -> L
# 
# Wherefrom the Lagrangian is a functional which takes in a field/function A, and returns a function
# on batched points [B, ...] that gives the values of the lagrangians of the field A at the given points
#       L : A -> l
#       l : [B, ...] -> [B]

# for typing purposes - a Lagrangian here will take in a Function and return a Function
Functional = Callable[[Function], Function]
# and a generator will take in a bunch of arguments and return a Lagrangian
FunctionalGenerator = Callable[..., Functional]


'''
# --- LAGRANGIAN VARIATION GENERATORS --- #

# these will be functionals that take in hyperparameters and metrics
# and return functionals that take in a bunch of time coordinates (single-variable)
# and return the lagrangian variation (i.e. lhs of the equations of motion s.t. rhs=0) at the given points

# for these, the input function is position, taken to be CONTRAVARIANT.
# should be a function like:
# x : [B] -> [B, N]

def FreeParticle(
    m : tf.Tensor, # mass of the free particle, should be a scalar

    geometry : Geometry   # geometry of our space (unused here)
) -> Functional:
      
    def _s(x : Function) -> Function:
        a = diff(diff(x))
        a = apply_fn(tf.squeeze, a)
        return scale_fn(m, a) 
    
    return _s 

def SpringParticle(
    m : tf.Tensor, # mass of the free particle, should be a scalar
    k : tf.Tensor, # spring constant k, should be a scalar

    geometry : Geometry # geometry of our space (unused here)
):
    
    def _s(x : Function) -> Function:
        a = diff(diff(x))
        a = apply_fn(tf.squeeze, a)

        F = scale_fn(m, a)
        kx = scale_fn(k, x)

        return add_fn(F, kx) # F + kx = 0
    
    return _s 
'''


# --- LAGRANGIAN DENSITY VARIATION GENERATORS --- #

# these will be functionals that take in hyperparameters and metrics
# and return functionals that take in a field and return the lagrangian density variation
# (i.e. lhs of eqs of motion s.t. rhs = 0) at the given field value

# here we multiply by volume form, since these are densities

# scalar field phi
def FreeScalarField(
    m : tf.Tensor = zero, # mass of the scalar field
    geometry : Geometry = Minkowski(N=4) # geometry should be a tuple of tensor functions (g, g_inv)
) -> Functional:
    
    g, g_inv = geometry
    vol_form = dVol(g)

    # phi should be a function [B, N] -> [B]
    if m == zero:
        def _s(phi : Function) -> Function:
            kinetic_term = Norm(grad(phi), g_inv)

            return mul_fn(vol_form, scale_fn(half, kinetic_term)) 

    else:
        def _s(phi : Function) -> Function:
            kinetic_term = Norm(grad(phi), g_inv) 
            mass_term = scale_fn(-m*m, mul_fn(phi, phi))

            return mul_fn(vol_form, scale_fn(half, add_fn(kinetic_term, mass_term))) 

    return _s


# free maxwell field A_mu
def FreeMaxwellField(
    geometry : Geometry = Minkowski(N=4) # geometry should be a tuple of tensor functions (g, g_inv)
):
    g, g_inv = geometry
    vol_form = dVol(g)

    def _s(A : Function) -> Function:
        # A should be a function [B, N] -> [B, N]

        f = grad(A) # [B, N, N] << last axis marks derivative variables
        g = apply_fn(tf.transpose, f, perm=(0, 2, 1))

        F = sub_fn(f, g) # F_mu nu = d_mu A_nu - d_nu A_mu

        kinetic_term = Norm(F, g_inv)

        return mul_fn(vol_form, scale_fn(-quarter, kinetic_term))

    return _s
        