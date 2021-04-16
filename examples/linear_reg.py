import miniml
from utils import *

# Adapted from:
# https://lucidar.me/en/neural-networks/simplest-neural-netwok-ever/

# init data
np.random.seed(3)
X = np.linspace(-10, 10, num=500)
Y = 0.6 * X + 2 + np.random.normal(size=500)

X_train = X.reshape((1, X.shape[0]))
Y_train = Y.reshape((1, Y.shape[0]))

# init params
rate = 1
epochs = 2000

# create model
model = miniml.Model(X_train.shape[0])
model.add(1, 'linear', 'plain')

# train model
optimizer = miniml.Optimizer(
    cost = 'mse',
    epochs = epochs,
    init_seed = 48,
    verbose = 500)

costs = optimizer.train_gd(model, X_train, Y_train, rate)

# plot results
plot_costs(costs, rate, epochs)
plot_regression(model, X_train, Y_train)
