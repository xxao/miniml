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
        
        self._cache = []
    
    
    @property
    def cache(self):
        """Gets current weights."""
        
        return self._cache
    
    
    def train(self, model, X, Y, lamb=0, **optimizer_params):
        """
        Trains model by data using specified optimizer.
        
        Args:
            model: Model
                Network model to train.
            
            X: np.ndarray
                Training dataset input.
            
            Y: np.ndarray
                Training dataset output.
            
            lamb: float
                L2 regularization lambda parameter.
            
            optimizer_params: {str:any}
                Specific optimizer params.
        
        Returns:
            costs: (float,)
                Intermediate costs.
        """
        
        costs = []
        batch_seed = self._batch_seed or 0
        
        # reset model params and cashes
        model.clear()
        
        # reset optimizer cache
        self._cache = [None]*len(model.layers)
        
        # initialize layers with given seed
        if self._init_seed:
            np.random.seed(self._init_seed)
            model.initialize(X[0].shape)
        
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
                Y_hat = model.forward(X_train)
                
                # compute cost
                cost, dA = self._calc_cost(model, Y_train, Y_hat, lamb=lamb)
                cost_total += cost
                
                # backward propagation
                model.backward(dA, lamb=lamb)
                
                # update params
                self.update(model, **optimizer_params)
            
            # average cost
            cost_avg = cost_total / len(batches)
            
            # remember cost
            if self._store and epoch % self._store == 0:
                costs.append(cost_avg)
            
            # print current cost
            if self._verbose and epoch % self._verbose == 0:
                print("Cost for epoch #%d: %.6f" % (epoch, cost_avg))
        
        return costs
    
    
    def update(self, model, **optimizer_params):
        """
        Updates params for all layers.
        
        Args:
            model: miniml.Model
                Model to update.
            
            optimizer_params: {str: any}
                Optimizer update parameters.
        """
        
        pass
    
    
    def _calc_cost(self, model, Y, Y_hat, lamb=0):
        """Calculates cost and its derivatives."""
        
        # compute cost
        cost, dA = self._cost(Y, Y_hat)
        
        # apply L2 regularization
        if lamb != 0:
            m = Y.shape[0]
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


class GradDescent(Optimizer):
    """Gradient descent optimizer."""
    
    
    def train(self, model, X, Y, rate=0.1, lamb=0):
        """
        Trains model by data using gradient descent.
        
        Args:
            model: miniml.Model
                Network model to train.
            
            X: np.ndarray
                Training dataset input.
            
            Y: np.ndarray
                Training dataset output.
            
            rate: float
                Learning rate.
            
            lamb: float
                L2 regularization lambda parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return super().train(
            model = model,
            X = X,
            Y = Y,
            lamb = lamb,
            rate = rate)
    
    
    def update(self, model, rate=0.1):
        """
        Updates params for all layers.
        
        Args:
            model: miniml.Model
                Model to update.
            
            rate: float
                Learning rate.
        """
        
        # update model
        for i, layer in enumerate(model.layers):
            
            # skip if not to optimize
            if not layer.OPTIMIZE:
                continue
            
            # update params
            W = layer.W - rate * layer.dW
            b = layer.b - rate * layer.db
            
            # update layer
            layer.update(W, b)


class Momentum(Optimizer):
    """Gradient descent optimizer with Momentum."""
    
    
    def train(self, model, X, Y, rate=0.1, lamb=0, beta=0.9):
        """
        Trains model by data using gradient descent with Momentum.
        
        Args:
            model: miniml.Model
                Network model to train.
            
            X: np.ndarray
                Training dataset input.
            
            Y: np.ndarray
                Training dataset output.
            
            rate: float
                Learning rate.
            
            lamb: float
                L2 regularization lambda parameter.
            
            beta: float
                Momentum parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return super().train(
            model = model,
            X = X,
            Y = Y,
            lamb = lamb,
            rate = rate,
            beta = beta)
    
    
    def update(self, model, rate=0.1, beta=0.9):
        """
        Updates params for all layers.
        
        Args:
            model: miniml.Model
                Model to update.
            
            rate: float
                Learning rate.
            
            beta:
                Momentum parameter.
        """
        
        # update model
        for i, layer in enumerate(model.layers):
            
            # skip if not to optimize
            if not layer.OPTIMIZE:
                continue
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {}
                vdW = np.zeros(layer.dW.shape)
                vdb = np.zeros(layer.db.shape)
            else:
                vdW = self.cache[i]['vdW']
                vdb = self.cache[i]['vdb']
            
            # update params
            vdW = beta * vdW + (1 - beta) * layer.dW
            vdb = beta * vdb + (1 - beta) * layer.db
            
            W = layer.W - rate * vdW
            b = layer.b - rate * vdb
            
            # update layer
            layer.update(W, b)
            
            # update cache
            self.cache[i]['vdW'] = vdW
            self.cache[i]['vdb'] = vdb


class RMSprop(Optimizer):
    """Gradient descent optimizer with RMSprop."""
    
    
    def train(self, model, X, Y, rate=0.1, lamb=0, beta=0.9):
        """
        Trains model by data using gradient descent with RMSprop.
        
        Args:
            model: miniml.Model
                Network model to train.
            
            X: np.ndarray
                Training dataset input.
            
            Y: np.ndarray
                Training dataset output.
            
            rate: float
                Learning rate.
            
            lamb: float
                L2 regularization lambda parameter.
            
            beta: float
                Momentum parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return super().train(
            model = model,
            X = X,
            Y = Y,
            lamb = lamb,
            rate = rate,
            beta = beta)
    
    
    def update(self, model, rate=0.1, beta=0.9, epsilon=1e-8):
        """
        Updates params for all layers.
        
        Args:
            model: miniml.Model
                Model to update.
            
            rate: float
                Learning rate.
            
            beta:
                Momentum parameter.
            
            epsilon: float
                Zero division corrector.
        """
        
        # update model
        for i, layer in enumerate(model.layers):
            
            # skip if not to optimize
            if not layer.OPTIMIZE:
                continue
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {}
                sdW = np.zeros(layer.dW.shape)
                sdb = np.zeros(layer.db.shape)
            else:
                sdW = self.cache[i]['sdW']
                sdb = self.cache[i]['sdb']
            
            # update params
            sdW = beta * sdW + (1 - beta) * layer.dW**2
            sdb = beta * sdb + (1 - beta) * layer.db**2
            
            W = layer.W - rate * layer.dW / np.sqrt(sdW + epsilon)
            b = layer.b - rate * layer.db / np.sqrt(sdb + epsilon)
            
            # update layer
            layer.update(W, b)
            
            # update cache
            self.cache[i]['sdW'] = sdW
            self.cache[i]['sdb'] = sdb


class Adam(Optimizer):
    """Gradient descent optimizer with Adam."""
    
    
    def train(self, model, X, Y, rate=0.1, lamb=0, beta1=0.9, beta2=0.999):
        """
        Trains model by data using gradient descent with Adam.
        
        Args:
            model: miniml.Model
                Network model to train.
            
            X: np.ndarray
                Training dataset input.
            
            Y: np.ndarray
                Training dataset output.
            
            rate: float
                Learning rate.
            
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
        
        return super().train(
            model = model,
            X = X,
            Y = Y,
            lamb = lamb,
            rate = rate,
            beta1 = beta1,
            beta2 = beta2)
    
    
    def update(self, model, rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Updates params for all layers.
        
        Args:
            model: miniml.Model
                Model to update.
            
            rate: float
                Learning rate.
            
            beta1: float
                First momentum parameter.
            
            beta2: float
                Second momentum parameter.
            
            epsilon: float
                Zero division corrector.
            
            epsilon: float
                Zero division corrector.
        """
        
        # update model
        for i, layer in enumerate(model.layers):
            
            # skip if not to optimize
            if not layer.OPTIMIZE:
                continue
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {}
                t = 1
                vdW = np.zeros(layer.dW.shape)
                vdb = np.zeros(layer.db.shape)
                sdW = np.zeros(layer.dW.shape)
                sdb = np.zeros(layer.db.shape)
            else:
                t = self.cache[i]['t'] + 1
                sdW = self.cache[i]['sdW']
                sdb = self.cache[i]['sdb']
                vdW = self.cache[i]['vdW']
                vdb = self.cache[i]['vdb']
            
            # update params
            vdW = beta1 * vdW + (1 - beta1) * layer.dW
            vdb = beta1 * vdb + (1 - beta1) * layer.db
            
            v_corr_dW = vdW / (1 - beta1**t)
            v_corr_db = vdb / (1 - beta1**t)
            
            sdW = beta2 * sdW + (1 - beta2) * layer.dW**2
            sdb = beta2 * sdb + (1 - beta2) * layer.db**2
            
            s_corr_dW = sdW / (1 - beta2**t)
            s_corr_db = sdb / (1 - beta2**t)
            
            W = layer.W - rate * v_corr_dW / (s_corr_dW**0.5 + epsilon)
            b = layer.b - rate * v_corr_db / (s_corr_db**0.5 + epsilon)
            
            # update layer
            layer.update(W, b)
            
            # update cache
            self.cache[i]['t'] = t
            self.cache[i]['vdW'] = vdW
            self.cache[i]['vdb'] = vdb
            self.cache[i]['sdW'] = sdW
            self.cache[i]['sdb'] = sdb


class Adagrad(Optimizer):
    """Gradient descent optimizer with Adagrad."""
    
    
    def train(self, model, X, Y, rate=0.1, lamb=0):
        """
        Trains model by data using gradient descent with Adagrad.
        
        Args:
            model: miniml.Model
                Network model to train.
            
            X: np.ndarray
                Training dataset input.
            
            Y: np.ndarray
                Training dataset output.
            
            rate: float
                Learning rate.
            
            lamb: float
                L2 regularization lambda parameter.
        
        Returns:
            costs:
                Intermediate costs.
        """
        
        return super().train(
            model = model,
            X = X,
            Y = Y,
            lamb = lamb,
            rate = rate)
    
    
    def update(self, model, rate=0.1, epsilon=1e-8):
        """
        Updates params for all layers.
        
        Args:
            model: miniml.Model
                Model to update.
            
            rate: float
                Learning rate.
            
            epsilon: float
                Zero division corrector.
        """
        
        # update model
        for i, layer in enumerate(model.layers):
            
            # skip if not to optimize
            if not layer.OPTIMIZE:
                continue
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {}
                sdW = np.zeros(layer.dW.shape)
                sdb = np.zeros(layer.db.shape)
            else:
                sdW = self.cache[i]['sdW']
                sdb = self.cache[i]['sdb']
            
            # update params
            sdW = sdW + layer.dW**2
            sdb = sdb + layer.db**2
            
            W = layer.W - rate * layer.dW / np.sqrt(sdW + epsilon)
            b = layer.b - rate * layer.db / np.sqrt(sdb + epsilon)
            
            # update layer
            layer.update(W, b)
            
            # update cache
            self.cache[i]['sdW'] = sdW
            self.cache[i]['sdb'] = sdb
