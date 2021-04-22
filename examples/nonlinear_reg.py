import miniml
import numpy as np

# Adapted from:
# https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/

# init data
np.random.seed(3)
X = np.linspace(-10, 10, num=1000)
Y = 0.1*X*np.cos(X) + 0.1*np.random.normal(size=1000)

X = X.reshape((len(X), 1))
Y = Y.reshape((len(Y), 1))

# create model
model = miniml.Model()
model.dense(1, 'linear', 'plain')
model.dense(64, 'relu', 'he')
model.dense(32, 'relu', 'he')
model.dense(1, 'linear', 'plain')

# init params
rate = 0.01
epochs = 1000

# train model
optimizer = miniml.Adam(
    cost = 'mse',
    epochs = epochs,
    init_seed = 48,
    store = 10,
    verbose = 200)

costs = optimizer.train(model, X, Y, rate)

# plot results
miniml.plot_costs(epochs, costs=costs)
miniml.plot_regression(model, X, Y)
