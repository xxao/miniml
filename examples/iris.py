import miniml
from sklearn.datasets import load_iris

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

# create model
model = miniml.Model()
model.dense(5, 'sigmoid', 'xavier')
model.dense(3, 'sigmoid', 'xavier')
model.dense(1, 'sigmoid', 'xavier')

# init params
rate = 1
epochs = 5000

# train model
# note that original implementation is without averaging across examples in LinearLayer.backward(...)
optimizer = miniml.GradDescent(
    cost = 'bce',
    epochs = epochs,
    init_seed = 48,
    verbose = 1000)

costs = optimizer.train(model, X, Y, rate)

# plot results
miniml.print_accuracy(model, X, Y)
miniml.plot_costs(epochs, costs=costs)
miniml.plot_boundaries(model, X, Y)
