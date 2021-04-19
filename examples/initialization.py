import sklearn.datasets
import miniml
from utils import *

# Adapted from DeepLearning.AI

# load data
np.random.seed(1)
X, Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
Y = Y.reshape((len(Y), 1))

# init params
rate = 0.01
epochs = 15000

# init model
model = miniml.Model()
model.dense(10, 'relu', 'he')
model.dense(5, 'relu', 'he')
model.dense(1, 'sigmoid', 'he')

# train model
optimizer = miniml.GradDescent(
    cost = 'bce',
    epochs = epochs,
    init_seed = 3,
    store = 1000)

costs = optimizer.train(model, X, Y, rate=rate)

# plot results
predict(model, X, Y)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X, Y)
