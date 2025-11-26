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



# --- LAGRANGIAN DENSITY VARIATION GENERATORS --- #

# these will be functionals that take in hyperparameters and metrics
# and return functionals that take in a field and return the lagrangian density variation
# (i.e. lhs of eqs of motion s.t. rhs = 0) at the given field value

# here we will have volume form built-in, since these are densities

# scalar field phi
def FreeScalarField(
    m : tf.Tensor = zero, # mass of the scalar field
    geometry : Geometry = Minkowski(N=4) # geometry should be a tuple of tensor functions (g, g_inv)
) -> Functional:
    
    g, g_inv = geometry
    vol_form = dVol(g)

    # phi should be a function [B, N] -> [B]
    if m == zero:
        def _s(phi : Function):
            return div(grad(phi), g_inv, vol_form)

    else:
        def _s(phi : Function) -> Function:
            laplace_term = div(grad(phi), g_inv, vol_form)
            mass_term = mul_fn(vol_form, scale_fn(m*m, phi))

            return add_fn(laplace_term, mass_term)

    return _s

