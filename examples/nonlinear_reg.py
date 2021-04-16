import miniml
from utils import *

# Adapted from:
# https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/

# init data
np.random.seed(3)
X = np.linspace(-10, 10, num=1000)
Y = 0.1*X*np.cos(X) + 0.1*np.random.normal(size=1000)

X_train = X.reshape((1, X.shape[0]))
Y_train = Y.reshape((1, Y.shape[0]))

# init params
rate = 0.01
epochs = 1000

# create model
model = miniml.Model(X_train.shape[0])
model.add(1, 'linear', 'plain')
model.add(64, 'relu', 'he')
model.add(32, 'relu', 'he')
model.add(1, 'linear', 'plain')

# train model
optimizer = miniml.Optimizer(
    cost = 'mse',
    epochs = epochs,
    init_seed = 48,
    store = 10,
    verbose = 200)

costs = optimizer.train_adam(model, X_train, Y_train, rate)

# plot results
plot_costs(costs, rate, epochs)
plot_regression(model, X_train, Y_train)
