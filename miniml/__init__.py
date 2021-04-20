#  Created byMartin.cz

# set version
version = (0, 1, 0)

# load modules
from . enums import *
from . costs import mean_squared_error, binary_cross_entropy, cross_entropy
from . activations import Activation, Linear, Sigmoid, ReLU, Tanh, Softmax
from . layer import Layer
from . conv import Conv2D
from . dense import Dense
from . dropout import Dropout
from . flatten import Flatten
from . pool import Pool, MaxPool, AvgPool
from . model import Model
from . optimize import Optimizer, GradDescent, Momentum, RMSprop, Adam, Adagrad
from . utils import shuffle_data, to_categorical, make_mini_batches

# load plots
try:
    from . plots import print_accuracy, plot_costs, plot_boundaries, plot_regression
except ImportError:
    pass
