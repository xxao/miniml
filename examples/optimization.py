import sklearn.datasets
import miniml
from utils import *

# Adapted from DeepLearning.AI

np.random.seed(3)
X_train, Y_train = sklearn.datasets.make_moons(n_samples=300, noise=.2)
X_train = X_train.T
Y_train = Y_train.reshape((1, Y_train.shape[0]))

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

# costs = optimizer.train_gd(model, X_train, Y_train, rate=rate)
# costs = optimizer.train_momentum(model, X_train, Y_train, rate=rate, beta=0.9)
# costs = optimizer.train_rmsprop(model, X_train, Y_train, rate=rate, beta=0.9)
costs = optimizer.train_adam(model, X_train, Y_train, rate=rate, beta1=0.9, beta2=0.999)
# costs = optimizer.train_adagrad(model, X_train, Y_train, rate=rate)

# plot results
predict(model, X_train, Y_train)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X_train, Y_train)
