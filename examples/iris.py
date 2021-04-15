from sklearn.datasets import load_iris
import miniml
from utils import *

# load data
iris = load_iris()

# take only sepal length (0th col) and sepal width (1st col)
X = iris.data[:, :2]

# fix the labels shape so that instead of (150,) its (150,1),
Y = iris.target.reshape((150, 1))

# shuffle data
X, Y = miniml.shuffle_data(X, Y, seed=48)

# transpose the data to correct shape for NN (n, m)
X_train = X.T
Y_train = Y.T

# make the target label virginica = 1 and the rest 0
Y_train = (Y_train == 2).astype('int')

# init params
rate = 1
epochs = 5000

# create model
model = miniml.Model(X_train.shape[0])
model.add(5, 'sigmoid', 'xavier')
model.add(3, 'sigmoid', 'xavier')
model.add(1, 'sigmoid', 'xavier')

# train the model
# note that original implementation is without averaging across examples in LinearLayer.backward(...)
optimizer = miniml.Optimizer(cost='bce', epochs=epochs, init_seed=48, verbose=1000)
costs = optimizer.train_gd(model, X_train, Y_train, rate)

# predict by model
predict(model, X_train, Y_train)
plot_costs(costs, rate, epochs)
plot_boundaries(model, X_train, Y_train)
