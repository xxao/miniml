import miniml
from sklearn.datasets import load_iris

# load data
iris = load_iris()
X = iris.data
y = iris.target

# convert to one-hot
Y, cats = miniml.to_categorical(y)
C = len(cats)

# shuffle data
X, Y = miniml.shuffle_data(X, Y, seed=48)

# create model
model = miniml.Model()
model.dense(5, 'relu', 'he')
model.dense(3, 'softmax', 'plain')

# init params
rate = 0.01
epochs = 5000
batch_size = 40

# train model
optimizer = miniml.Adam(
    cost = 'ce',
    epochs = epochs,
    init_seed = 48,
    verbose = 1000)

costs = optimizer.train(model, X, Y, rate)

# plot results
miniml.print_accuracy(model, X, Y)
miniml.plot_costs(epochs, costs=costs)
