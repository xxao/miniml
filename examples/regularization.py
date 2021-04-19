import scipy.io
import miniml
from utils import *

# Adapted from DeepLearning.AI

# load data
data = scipy.io.loadmat('../datasets/data.mat')
X = data['X']
Y = data['y']

# init params
rate = 0.3
epochs = 30000
lamb = 0
keep = 0.86

# init model
model = miniml.Model()
model.add(20, 'relu', 'xavier')
model.add(3, 'relu', 'xavier')
model.add(1, 'sigmoid', 'xavier')

# train model
optimizer = miniml.GradDescent(
    cost = 'bce',
    epochs = epochs,
    init_seed = 3,
    dropout_seed = 1,
    store = 1000,
    verbose = 10000)

costs = optimizer.train(model, X, Y,
    rate = rate,
    keep = keep,
    lamb = lamb)

# plot results
predict(model, X, Y)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X, Y)
