#  Created byMartin.cz

import numpy as np
from . enums import *
from . costs import *
from . utils import *


class Optimizer(object):
    """Provides a tool to initialize and perform model training."""
    
    
    def __init__(self, cost=MSE, epochs=10000, batch_size=0, batch_seed=None, init_seed=None, dropout_seed=None, store=100, verbose=1000):
        """
        Initialize a new instance of Optimizer.
        
        Args:
            cost: str
                Cost function name such as 'mse' or 'bce'.
            
            epochs: int
                Number of training epochs.
            
            batch_size: int
                Size of a mini-batch. Should be 64, 128, 256 or 512. If set to
                zero, no mini-batches will be created.
            
            batch_seed: int or None
                Random seed for mini-batches creation.
            
            init_seed: int or None
                Random seed for layer params.
            
            dropout_seed: int or None:
                Random seed for dropout.
            
            store: int
                Specifies cost storage interval.
            
            verbose: int
                Specifies cost printout interval. If set to zero, cost printout
                will be disabled.
        """
        
        self._cost = self._init_cost(cost)
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._batch_seed = batch_seed
        self._init_seed = init_seed
        self._dropout_seed = dropout_seed
        self._store = int(store)
        self._verbose = int(verbose)
    
    
    def train_model(self, model, X, Y, keep=1, lamb=0, optimizer=GD, **optimizer_params):
        """
        Trains model by data using specified optimizer.
        
        Args:
            model: Model
                Network model to train.
            
            X:
                Training data set input.
            
            Y:
                Training data set output.
            
            keep: float
                Dropout keep probability (0-1).
            
            lamb: float
                L2 regularization lambda parameter.
            
            optimizer: str
                Optimizer name such as 'gd', 'momentum', 'rmsprop',
                'adagrad' or 'adam'.
            
            optimizer_params: {str:any}
                Specific optimizer params.
        
        Returns:
            costs: (float,)
                Intermediate costs.
        """
        
        costs = []
        batch_seed = self._batch_seed or 0
        
        # set layers init seed
        if self._init_seed:
            np.random.seed(self._init_seed)
        
        # init layers params
        model.initialize(optimizer)
        
        # train model
        for epoch in range(self._epochs):
            
            # init mini-batches
            batch_seed = batch_seed + 1
            batches = make_mini_batches(X, Y, self._batch_size, batch_seed)
            
            # set layers dropout seed
            if self._dropout_seed:
                np.random.seed(self._dropout_seed)
            
            # init cost
            cost_total = 0
            
            # train model
            for X_train, Y_train in batches:
                
                # forward propagation
                Y_hat = model.forward(X_train, keep=keep)
                
                # compute cost
                cost, dA = self._calc_cost(model, Y_train, Y_hat, lamb=lamb)
                cost_total += cost
                
                # backward propagation
                model.backward(dA, lamb=lamb)
                
                # update params
                model.update(optimizer, **optimizer_params)
            
            # average cost
            cost_avg = cost_total / len(batches)
            
            # remember cost
            if self._store and epoch % self._store == 0:
                costs.append(cost_avg)
            
            # print current cost
            if self._verbose and epoch % self._verbose == 0:
                print("Cost for epoch #%d: %.6f" % (epoch, cost_avg))
        
        return costs
    
    
    def train_gd(self, model, X, Y, rate=0.1, keep=1, lamb=0):
        """
        Trains model by data using gradient descent.
        
        Args:
            model: Model
                Network model to train.
            
            X:
                Training data set input.
            
            Y:
                Training data set output.
            
            rate: float
                Learning rate.
            
            keep: float
                Dropout keep probability (0-1).
            
            lamb: float
                L2 regularization lambda parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return self.train_model(
            model = model,
            X = X,
            Y = Y,
            keep = keep,
            lamb = lamb,
            optimizer = GD,
            rate = rate)
    
    
    def train_momentum(self, model, X, Y, rate=0.1, keep=1, lamb=0, beta=0.9):
        """
        Trains model by data using gradient descent with Momentum.
        
        Args:
            model: Model
                Network model to train.
            
            X:
                Training data set input.
            
            Y:
                Training data set output.
            
            rate: float
                Learning rate.
            
            keep: float
                Dropout keep probability (0-1).
            
            lamb: float
                L2 regularization lambda parameter.
            
            beta: float
                Momentum parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return self.train_model(
            model = model,
            X = X,
            Y = Y,
            keep = keep,
            lamb = lamb,
            optimizer = MOMENTUM,
            rate = rate,
            beta = beta)
    
    
    def train_rmsprop(self, model, X, Y, rate=0.1, keep=1, lamb=0, beta=0.9):
        """
        Trains model by data using gradient descent with RMSprop.
        
        Args:
            model: Model
                Network model to train.
            
            X:
                Training data set input.
            
            Y:
                Training data set output.
            
            rate: float
                Learning rate.
            
            keep: float
                Dropout keep probability (0-1).
            
            lamb: float
                L2 regularization lambda parameter.
            
            beta: float
                Momentum parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return self.train_model(
            model = model,
            X = X,
            Y = Y,
            keep = keep,
            lamb = lamb,
            optimizer = RMSPROP,
            rate = rate,
            beta = beta)
    
    
    def train_adam(self, model, X, Y, rate=0.1, keep=1, lamb=0, beta1=0.9, beta2=0.999):
        """
        Trains model by data using gradient descent with Adam.
        
        Args:
            model: Model
                Network model to train.
            
            X:
                Training data set input.
            
            Y:
                Training data set output.
            
            rate: float
                Learning rate.
            
            keep: float
                Dropout keep probability (0-1).
            
            lamb: float
                L2 regularization lambda parameter.
            
            beta1: float
                First momentum parameter.
            
            beta2: float
                Second momentum parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return self.train_model(
            model = model,
            X = X,
            Y = Y,
            keep = keep,
            lamb = lamb,
            optimizer = ADAM,
            rate = rate,
            beta1 = beta1,
            beta2 = beta2)
    
    
    def train_adagrad(self, model, X, Y, rate=0.1, keep=1, lamb=0):
        """
        Trains model by data using gradient descent with Adagrad.
        
        Args:
            model: Model
                Network model to train.
            
            X:
                Training data set input.
            
            Y:
                Training data set output.
            
            rate: float
                Learning rate.
            
            keep: float
                Dropout keep probability (0-1).
            
            lamb: float
                L2 regularization lambda parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return self.train_model(
            model = model,
            X = X,
            Y = Y,
            keep = keep,
            lamb = lamb,
            optimizer = ADAGRAD,
            rate = rate)
    
    
    def _calc_cost(self, model, Y, Y_hat, lamb=0):
        """Calculates cost and its derivatives."""
        
        # compute cost
        cost, dA = self._cost(Y, Y_hat)
        
        # apply L2 regularization
        if lamb != 0:
            m = Y.shape[1]
            Ws = sum(np.sum(np.square(layer.W)) for layer in model.layers)
            L2_cost = (lamb / (2 * m)) * Ws
            cost += L2_cost
        
        return cost, dA
    
    
    def _init_cost(self, name):
        """Initialize cost function."""
        
        if name == MSE:
            return mean_squared_error
        
        elif name == BCE:
            return binary_cross_entropy
        
        elif name == CE:
            return cross_entropy
        
        raise ValueError("Unknown cost function specified! -> '%s" % name)

