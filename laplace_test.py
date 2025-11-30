# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Laplacian Test

from operators import *
from keras.models import save_model, load_model
from keras.optimizers import AdamW
from networks import SpatialKernel, OperatorWrapper
from geometry import Euclidean
from distributions import Gaussian, Reciprocal
from system import System
from plots import save_heatmap

MODELPATH = None
NAME = 'test4'

# we test our current abilities.
# i'll try to find the derivative and second derivative of the square function
# using our mesh+convolution framework.
step = 0.05
start = -4.8 # the size of the mesh should be divisible by 2 at least thrice
end = 4.8

# we want each point to look like [B, 1]
dX = tf.constant([step, step])
ranges = tf.constant([[start, start], [end, end]])
domain = Domain(Euclidean(2), ranges, dX) # this is now [b, b, 1])

base_image = Image(domain, pad=32)

# we now try training over many different forcing terms
forcing_terms = [
    Reciprocal([-0.5, 0.0], scale=one),
    Gaussian([-1.2, 0.4], 0.0125, scale=one),

    Reciprocal([0.2, -1.4], scale=two),
    Gaussian([-0.3, 1.6], 0.0175, scale=two),

    Reciprocal([-0.8, 1.0], scale=half),
    Gaussian([0.9, 0.7], 0.0225, scale=half),

    Reciprocal([-0.5, -2.9], scale=one),
    Gaussian([3.1, 2.5], 0.0275, scale=two),
]


if MODELPATH is None:
    system = System(2, operator=FlatLaplacian, pointwise_loss=tf.square)

    model = SpatialKernel(shape=[], dims=2, size=11, activation='relu')

    for f in forcing_terms:
        system.force(f)
        system.train(model, domain, AdamW(), epochs=25)
else:
    model = load_model(MODELPATH)

solution_operator = OperatorWrapper(model)

i = 0
for forcing_term in forcing_terms:
    # we save the forcing image
    forcing_image = forcing_term(base_image)
    save_heatmap(forcing_image.view(), f'logs/laplace/{NAME}/imgs/{i}-forcing_term.png')

    # the learned solution
    solution_image = solution_operator(forcing_image)
    save_heatmap(solution_image.view(), f'logs/laplace/{NAME}/imgs/{i}-solution.png')

    # and the "reported" forcing term - i.e. what we get
    # when we apply the desired operator on the learned solution
    laplacian_image = FlatLaplacian(solution_image)
    save_heatmap(laplacian_image.view(), f'logs/laplace/{NAME}/imgs/{i}-reported_forcing_term.png')

    i+=1

if MODELPATH is None:
    save_model(model, f'logs/laplace/{NAME}/model.keras')