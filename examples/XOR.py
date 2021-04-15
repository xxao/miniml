import miniml
from utils import *

# Test case from
# https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html

# init data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

X_train = X.T
Y_train = Y.T

# init params
rate = 1
epochs = 20000

# create model
model = miniml.Model(X_train.shape[0])
model.add(5, 'sigmoid', 'xavier')
model.add(3, 'sigmoid', 'xavier')
model.add(1, 'sigmoid', 'xavier')

# train the model
# note that original implementation is without averaging across examples in LinearLayer.backward(...)
optimizer = miniml.Optimizer(cost='mse', epochs=epochs, init_seed=48)
costs = optimizer.train_gd(model, X_train, Y_train, rate)

# predict by model
predict(model, X_train, Y_train)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X_train, Y_train)
