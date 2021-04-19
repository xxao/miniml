import miniml
from utils import *

# Adapted from:
# http://saitcelebi.com/tut/output/part2.html

# init data
X = np.array([
    [-0.1, 1.4],
    [-0.5, 0.2],
    [1.3, 0.9],
    [-0.6, 0.4],
    [-1.6, 0.2],
    [0.2, 0.2],
    [-0.3, -0.4],
    [0.7, -0.8],
    [1.1, -1.5],
    [-1.0, 0.9],
    [-0.5, 1.5],
    [-1.3, -0.4],
    [-1.4, -1.2],
    [-0.9, -0.7],
    [0.4, -1.3],
    [-0.4, 0.6],
    [0.3, -0.5],
    [-1.6, -0.7],
    [-0.5, -1.4],
    [-1.0, -1.4]])

y = np.array([0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 2, 1, 0, 1, 2, 2, 2])

# convert to one-hot
Y, cats = miniml.to_categorical(y)
C = len(cats)

# init params
rate = 2
epochs = 40

# create model
model = miniml.Model()
# model.add(32, 'relu', 'he')
model.add(C, 'softmax', 'plain')

# train model
optimizer = miniml.Optimizer(
    cost = 'ce',
    epochs = epochs,
    init_seed = 48,
    store = 1,
    verbose = 10)

costs = optimizer.train_gd(model, X, Y, rate)

# plot results
predict(model, X, Y)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X, Y)
