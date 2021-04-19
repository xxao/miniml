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
model.add(5, 'relu', 'he')
model.add(2, 'relu', 'he')
model.add(1, 'sigmoid', 'he')

# train model
optimizer = miniml.Optimizer(
    cost = 'bce',
    epochs = epochs,
    batch_size = batch_size,
    batch_seed = 10,
    init_seed = 3,
    store = 1000)

# costs = optimizer.train_gd(model, X, Y, rate=rate)
# costs = optimizer.train_momentum(model, X, Y, rate=rate, beta=0.9)
# costs = optimizer.train_rmsprop(model, X, Y, rate=rate, beta=0.9)
costs = optimizer.train_adam(model, X, Y, rate=rate, beta1=0.9, beta2=0.999)
# costs = optimizer.train_adagrad(model, X, Y, rate=rate)

# plot results
predict(model, X, Y)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X, Y)
