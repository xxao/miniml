from sklearn.datasets import load_iris
import miniml
from utils import *

# load data
iris = load_iris()
X = iris.data
y = iris.target

# convert to one-hot
Y, cats = miniml.to_categorical(y)
C = len(cats)

# shuffle data
X, Y = miniml.shuffle_data(X, Y, seed=48)

# transpose to correct shape for NN (n, m)
X_train = X.T
Y_train = Y.T

# init params
rate = 0.01
epochs = 5000
batch_size = 40

# create model
model = miniml.Model()
model.add(5, 'relu', 'he')
model.add(3, 'softmax', 'plain')

# train model
optimizer = miniml.Optimizer(
    cost = 'ce',
    epochs = epochs,
    init_seed = 48,
    verbose = 1000)

costs = optimizer.train_adam(model, X_train, Y_train, rate)

# plot results
plot_costs(costs, rate, epochs)
