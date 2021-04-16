import sklearn.datasets
import miniml
from utils import *

# Adapted from DeepLearning.AI

# load data
np.random.seed(1)
X_train, Y_train = sklearn.datasets.make_circles(n_samples=300, noise=.05)
X_train = X_train.T
Y_train = Y_train.reshape((1, Y_train.shape[0]))

# init params
rate = 0.01
epochs = 15000

# init model
model = miniml.Model(X_train.shape[0])
model.add(10, 'relu', 'he')
model.add(5, 'relu', 'he')
model.add(1, 'sigmoid', 'he')

# train model
optimizer = miniml.Optimizer(
    cost = 'bce',
    epochs = epochs,
    init_seed = 3,
    store = 1000)

costs = optimizer.train_gd(model, X_train, Y_train, rate=rate)

# plot results
predict(model, X_train, Y_train)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X_train, Y_train)
