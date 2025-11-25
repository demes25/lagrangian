# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Geometry

from functionals import *


# some geometric and notions such as inner products and norms, volume forms and such

# U, V both [B, n * N]
# returns inner product [B]
def InnerProduct(U, V, g=None):
    rank_plus_1 = tf.rank(U) # rank, plus 1 to account for the batch dimension
    
    if g is None:
        # if g is none, we assume that we want straight-up dot products
        # so do exactly that
        prod = mul_fn(U, V)
        return apply_fn(tf.reduce_sum, prod, axis=-tf.range(1, rank_plus_1))
    
    rank = rank_plus_1 - 1

    if rank == 0:
        # just multiply if we only have scalars
        return mul_fn(U, V)
    
    # now for interesting cases
    if rank == 1:
        # for vectors: we want einstein summation
        def _bilinear(u, a, v):
            # we write einstein summation, where b is the batch index and ij are dummy
            return tf.einsum('bi,bij,bj->b', u, a, v)
        
    elif rank == 2:
        # for tensors: we want einstein summation through all indices
        def _bilinear(u, a, v):
            # we write einstein summation, where b is the batch index and IJ/ij are dummy
            return tf.einsum('bIJ,bIi,bJj,bij->b', u, a, a, v)
    
    else:
        #TODO: generalize to rank n
        raise NotImplementedError('Arbitrary rank not yet implemented.')
    
    return apply_fn(_bilinear, U, g, V)

# returns InnerProduct(U, U)
def Norm(U, g=None):
    return InnerProduct(U, U, g=g)


# volume form - sqrt(abs(det(g)))
def dVol(g):  
    abs_detg = apply_fn(tf.abs, det(g))
    return apply_fn(tf.sqrt, abs_detg)




# some generic tensors:

# identity tensor
def Delta(N):
    return tf.linalg.diag(tf.ones([N], dtype=DTYPE))

# minkowski tensor - signature (-, +, +, +, ...)
def Eta(N):
    return tf.linalg.diag(tf.constant([-1] + [1]*(N-1), dtype=DTYPE))

