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
        
        # reset optimizer cache
        self._cache = [None]*len(model.layers)
        
        # set initialization seed
        if self._init_seed:
            np.random.seed(self._init_seed)
        
        # initialize layers params and cashes
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
            losses = sum(layer.loss() for layer in model.layers)
            L2_cost = (lamb / (2 * m)) * losses
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
            
            # check if trainable
            if not layer.trainable:
                continue
            
            # get params and gradients
            params = layer.parameters()
            grads = layer.gradients()
            
            # update params
            updates = []
            for p in range(len(params)):
                param = params[p] - rate * grads[p]
                updates.append(param)
            
            # update layer
            layer.update(*updates)


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
            
            # check if trainable
            if not layer.trainable:
                continue
            
            # get params and gradients
            params = layer.parameters()
            grads = layer.gradients()
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {'v': [None]*len(params)}
            
            # update params
            updates = []
            for p in range(len(params)):
                
                # get layer parameter and its gradient
                param = params[p]
                grad = grads[p]
                
                # get or init gradient v
                v_grad = self.cache[i]['v'][p]
                if v_grad is None:
                    v_grad = np.zeros(grad.shape)
                
                # update param
                v_grad = beta * v_grad + (1 - beta) * grad
                param = param - rate * v_grad
                updates.append(param)
                
                # update cache
                self.cache[i]['v'][p] = v_grad
            
            # update layer
            layer.update(*updates)


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
            
            # check if trainable
            if not layer.trainable:
                continue
            
            # get params and gradients
            params = layer.parameters()
            grads = layer.gradients()
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {'s': [None]*len(params)}
            
            # update params
            updates = []
            for p in range(len(params)):
                
                # get layer parameter and its gradient
                param = params[p]
                grad = grads[p]
                
                # get or init gradient s
                s_grad = self.cache[i]['s'][p]
                if s_grad is None:
                    s_grad = np.zeros(grad.shape)
                
                # update param
                s_grad = beta * s_grad + (1 - beta) * grad**2
                param = param - rate * grad / np.sqrt(s_grad + epsilon)
                updates.append(param)
                
                # update cache
                self.cache[i]['s'][p] = s_grad
            
            # update layer
            layer.update(*updates)


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
            
            # check if trainable
            if not layer.trainable:
                continue
            
            # get params and gradients
            params = layer.parameters()
            grads = layer.gradients()
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {
                    't': 0,
                    'v': [None]*len(params),
                    's': [None]*len(params)}
            
            # update and get cycle
            self.cache[i]['t'] += 1
            t = self.cache[i]['t']
            
            # update params
            updates = []
            for p in range(len(params)):
                
                # get layer parameter and its gradient
                param = params[p]
                grad = grads[p]
                
                # get or init gradient v
                v_grad = self.cache[i]['v'][p]
                if v_grad is None:
                    v_grad = np.zeros(grad.shape)
                
                # get or init gradient s
                s_grad = self.cache[i]['s'][p]
                if s_grad is None:
                    s_grad = np.zeros(grad.shape)
                
                # update param
                v_grad = beta1 * v_grad + (1 - beta1) * grad
                v_corr = v_grad / (1 - beta1**t)
                
                s_grad = beta2 * s_grad + (1 - beta2) * grad**2
                s_corr = s_grad / (1 - beta2**t)
                
                param = param - rate * v_corr / (s_corr**0.5 + epsilon)
                updates.append(param)
                
                # update cache
                self.cache[i]['v'][p] = v_grad
                self.cache[i]['s'][p] = s_grad
            
            # update layer
            layer.update(*updates)


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
            
            # check if trainable
            if not layer.trainable:
                continue
            
            # get params and gradients
            params = layer.parameters()
            grads = layer.gradients()
            
            # init cache
            if self.cache[i] is None:
                self.cache[i] = {'s': [None]*len(params)}
            
            # update params
            updates = []
            for p in range(len(params)):
                
                # get layer parameter and its gradient
                param = params[p]
                grad = grads[p]
                
                # get or init gradient s
                s_grad = self.cache[i]['s'][p]
                if s_grad is None:
                    s_grad = np.zeros(grad.shape)
                
                # update param
                s_grad = s_grad + grad**2
                param = param - rate * grad / np.sqrt(s_grad + epsilon)
                updates.append(param)
                
                # update cache
                self.cache[i]['s'][p] = s_grad
            
            # update layer
            layer.update(*updates)
