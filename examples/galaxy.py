import miniml
import numpy as np
from matplotlib import pyplot as plt

# Adapted from:
# https://github.com/kevinzakka/cs231n.github.io/blob/master/neural-networks-case-study.md

# init data
N = 100  # points per class
C = 3  # classes

X = np.zeros((N*C, 2))
y = np.zeros(N*C, dtype='uint8')

np.random.seed(3)
for j in range(C):
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2
    ixd = range(N*j, N*(j+1))
    X[ixd] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ixd] = j

# convert to one-hot
Y, cats = miniml.to_categorical(y)

# create model
model = miniml.Model()
model.dense(16, 'relu', 'he')
model.dense(C, 'softmax', 'plain')

# init params
rate = 1
epochs = 1000

# train model
optimizer = miniml.GradDescent(
    cost = 'ce',
    epochs = epochs,
    init_seed = 48,
    store = 100,
    verbose = 200)

costs = optimizer.train(model, X, Y, rate)

# plot results
miniml.print_accuracy(model, X, Y)
miniml.plot_costs(epochs, costs=costs)
miniml.plot_boundaries(model, X, Y)
