import miniml
import numpy as np

# Adapted from:
# https://lucidar.me/en/neural-networks/simplest-neural-netwok-ever/

# init data
np.random.seed(3)
X = np.linspace(-10, 10, num=500)
Y = 0.6 * X + 2 + np.random.normal(size=500)

X = X.reshape((len(X), 1))
Y = Y.reshape((len(Y), 1))

# init params
rate = 1
epochs = 2000

# create model
model = miniml.Model()
model.dense(1, 'linear', 'plain')

# train model
optimizer = miniml.GradDescent(
    cost = 'mse',
    epochs = epochs,
    init_seed = 48,
    verbose = 500)

costs = optimizer.train(model, X, Y, rate)

# plot results
miniml.plot_costs(epochs, costs=costs)
miniml.plot_regression(model, X, Y)
