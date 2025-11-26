# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Geometry

from functionals import *


# some geometric and notions such as inner products and norms, volume forms and such

# U, V both [B, n * N]
# returns inner product [B]
def InnerProduct(U : Function, V : Function, g : Function | None = None) -> Function:
    if g is None:
        def _s(x):
            # if g is none, we assume that we want straight-up dot products
            # so do exactly that
            prod = U(x) * V(x)
            return tf.reduce_sum(prod, axis=-tf.range(1, tf.rank(prod)))
    
    else:
        def _s(x):
            ux = U(x)
            vx = V(x)
            rank = tf.rank(ux) - 1

            if rank == 0:
                # just multiply if we only have scalars
                return ux * vx
            
            gx = g(x)

            # now for interesting cases
            if rank == 1:
                # for vectors: we want einstein summation
                # we write einstein summation, where b is the batch index and ij are dummy
                return tf.einsum('bi,bij,bj->b', ux, gx, vx)
                
            elif rank == 2:
                # for tensors: we want einstein summation through all indices
                # we write einstein summation, where b is the batch index and IJ/ij are dummy
                return tf.einsum('bIJ,bIi,bJj,bij->b', ux, gx, gx, vx)
            
            else:
                #TODO: generalize to rank n
                raise NotImplementedError('Arbitrary rank not yet implemented.')

    return _s 


# returns InnerProduct(U, U)
def Norm(U : Function, g : Function | None = None) -> Function:
    return InnerProduct(U, U, g=g)


# volume form - sqrt(abs(det(g)))
def dVol(g : Function | None = None) -> Function:  
    if g is None:
        # g None assumes Euclidean metric.
        return constant_fn(one)
    else:
        abs_detg = apply_fn(tf.abs, det(g))
        return apply_fn(tf.sqrt, abs_detg)



# some generic tensors:

# identity tensor
def Delta(N) -> tf.Tensor:
    return tf.linalg.diag(tf.ones([N], dtype=DTYPE))

# minkowski tensor - signature (-, +, +, +, ...)
def Eta(N) -> tf.Tensor:
    return tf.linalg.diag(tf.constant([-1] + [1]*(N-1), dtype=DTYPE))

