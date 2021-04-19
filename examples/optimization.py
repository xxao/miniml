import sklearn.datasets
import miniml
from utils import *

# Adapted from DeepLearning.AI

np.random.seed(3)
X, Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
Y = Y.reshape((len(Y), 1))

# init params
rate = 0.0007
epochs = 10000
batch_size = 64

# init model
model = miniml.Model()
model.dense(5, 'relu', 'he')
model.dense(2, 'relu', 'he')
model.dense(1, 'sigmoid', 'he')

# train model (try different optimizers)
optimizer = miniml.Adam(
    cost = 'bce',
    epochs = epochs,
    batch_size = batch_size,
    batch_seed = 10,
    init_seed = 3,
    store = 1000)

costs = optimizer.train(model, X, Y, rate)

# plot results
predict(model, X, Y)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X, Y)
