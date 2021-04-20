import miniml
import numpy as np

# Adapted from:
# https://github.com/RafayAK/NothingButNumPy/blob/master/Understanding_and_Creating_NNs/3_layer_toy_network_XOR.ipynb

# init data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# init params
rate = 1
epochs = 20000

# create model
model = miniml.Model()
model.dense(5, 'sigmoid', 'xavier')
model.dense(3, 'sigmoid', 'xavier')
model.dense(1, 'sigmoid', 'xavier')

# train model
# note that original implementation is without averaging across examples in LinearLayer.backward(...)
optimizer = miniml.GradDescent(
    cost = 'mse',
    epochs = epochs,
    init_seed = 48)

costs = optimizer.train(model, X, Y, rate)

# plot results
miniml.print_accuracy(model, X, Y)
miniml.plot_costs(epochs, costs=costs)
miniml.plot_boundaries(model, X, Y)
