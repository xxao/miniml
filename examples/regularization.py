import miniml
import scipy.io

# Adapted from DeepLearning.AI

# load data
data = scipy.io.loadmat('../datasets/regularization.mat')
X = data['X']
Y = data['y']

# init params
rate = 0.3
epochs = 30000
lamb = 0.7

# init model
model = miniml.Model()
model.dense(20, 'relu', 'xavier')
model.dense(3, 'relu', 'xavier')
model.dense(1, 'sigmoid', 'xavier')

# train model
optimizer = miniml.GradDescent(
    cost = 'bce',
    epochs = epochs,
    init_seed = 3,
    dropout_seed = 1,
    store = 1000,
    verbose = 10000)

costs = optimizer.train(model, X, Y, rate, lamb=lamb)

# plot results
miniml.print_accuracy(model, X, Y)
miniml.plot_costs(epochs, costs=costs)
miniml.plot_boundaries(model, X, Y)
