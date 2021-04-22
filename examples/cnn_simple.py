import miniml
import h5py
import numpy as np

# Adapted from DeepLearning.AI

# load data
train_dataset = h5py.File('../datasets/train_happy.h5', "r")
X_train = np.array(train_dataset["train_set_x"][:]).astype('float32')/255.
Y_train = np.array(train_dataset["train_set_y"][:]).reshape(-1, 1)

test_dataset = h5py.File('../datasets/test_happy.h5', "r")
X_test = np.array(test_dataset["test_set_x"][:]).astype('float32')/255.
Y_test = np.array(test_dataset["test_set_y"][:]).reshape(-1, 1)

# create model
model = miniml.Model()
model.conv2d(8, ksize=7, stride=1, activation='tanh')
model.flatten()
model.dense(1, 'sigmoid', 'plain')

# init params
rate = 0.001
epochs = 10

optimizer = miniml.Adam(
    cost = 'bce',
    epochs = epochs,
    batch_size = 16,
    batch_seed = 3,
    init_seed = 42,
    store = 1,
    verbose = 1)

costs = optimizer.train(model, X_train, Y_train, rate)

# plot results
miniml.print_accuracy(model, X_train, Y_train)
miniml.print_accuracy(model, X_test, Y_test, label="Test")
miniml.plot_costs(epochs, costs=costs)

