import miniml
import sklearn.datasets
import numpy as np

# Adapted from DeepLearning.AI

# load data
np.random.seed(1)
X, Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
Y = Y.reshape((len(Y), 1))

# init model
model = miniml.Model()
model.dense(10, 'relu', 'he')
model.dense(5, 'relu', 'he')
model.dense(1, 'sigmoid', 'he')

# init params
rate = 0.01
epochs = 15000

# train model
optimizer = miniml.GradDescent(
    cost = 'bce',
    epochs = epochs,
    init_seed = 3,
    store = 1000)

costs = optimizer.train(model, X, Y, rate=rate)

# plot results
miniml.print_accuracy(model, X, Y)
miniml.plot_costs(epochs, costs=costs)
miniml.plot_boundaries(model, X, Y)
