from sklearn.datasets import load_iris
import miniml
from utils import *

# Adapted from:
# https://github.com/RafayAK/NothingButNumPy/blob/master/Understanding_and_Creating_Binary_Classification_NNs/3_layer_toy_neural_network_on_iris_sepals.ipynb

# load data
iris = load_iris()

# take only sepal length (0th col) and sepal width (1st col)
X = iris.data[:, :2]

# fix the labels shape so that instead of (150,) its (150,1),
Y = iris.target.reshape((150, 1))

# shuffle data
X, Y = miniml.shuffle_data(X, Y, seed=48)

# make the target label virginica = 1 and the rest 0
Y = (Y == 2).astype('int')

# init params
rate = 1
epochs = 5000

# create model
model = miniml.Model()
model.add(5, 'sigmoid', 'xavier')
model.add(3, 'sigmoid', 'xavier')
model.add(1, 'sigmoid', 'xavier')

# train model
# note that original implementation is without averaging across examples in LinearLayer.backward(...)
optimizer = miniml.GradDescent(
    cost = 'bce',
    epochs = epochs,
    init_seed = 48,
    verbose = 1000)

costs = optimizer.train(model, X, Y, rate)

# plot results
predict(model, X, Y)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X, Y)
