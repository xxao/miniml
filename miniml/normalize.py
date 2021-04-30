#  Created byMartin.cz

import numpy as np
from . layer import Layer


class BatchNorm(Layer):
    """Represents a batch normalization layer of neural network."""
    
    
    def __init__(self, momentum=0.99, epsilon=1e-8):
        """Initializes a new instance of BatchNorm."""
        
        self._epsilon = float(epsilon)
        self._momentum = float(momentum)
        
        self._gamma = None
        self._beta = None
        self._dgamma = None
        self._dbeta = None
        self._mean = None
        self._variance = None
    
    
    def initialize(self, shape):
        """
        Clears caches and re-initializes params.
        
        Args:
            shape: (int,)
                Expected input shape. The shape must be provided without first
                dimension for number of samples (m).
        
        Returns:
            (int,)
                Output shape. The shape is provided without first dimension for
                number of samples (m).
        """
        
        # clear params and caches
        self._gamma = None
        self._beta = None
        self._dgamma = None
        self._dbeta = None
        self._mean = None
        self._variance = None
        
        # init params
        self._init_params(shape)
        
        # return output shape
        return self.outshape(shape)
    
    
    def forward(self, X, training, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is either (m, n) or (m, n_h, n_w, n_c).
            
            training: bool
                If set to True, the input data/activations are considered as
                training set.
        
        Returns:
            Flattened data into (m, n) or (m, n_h x n_w x n_c), depending on the
            input shape.
        """
        
        self._X = X
        
        # init params
        if self._gamma is None:
            self._init_params(X.shape[1:])
        
        # get mean and variance
        if training:
            axis = tuple(range(X.ndim-1))
            mean = np.mean(X, axis=axis)
            variance = np.mean((X - mean)**2, axis=axis)
         
        else:
            mean = self._mean
            variance = self._variance
        
        # normalize
        var_sqrt = np.sqrt(variance + self._epsilon)
        X_mean = (X - mean)
        X_hat = X_mean / var_sqrt
        
        # scale and shift
        A = self._gamma * X_hat + self._beta
        
        # update running mean and variance
        if training:
            self._mean = self._mean * self._momentum + (1 - self._momentum) * mean
            self._variance = self._variance * self._momentum + (1 - self._momentum) * variance
            
            var_sqrt = np.sqrt(self._variance + self._epsilon)
            #X_mean = (X - self._mean)
            X_hat = X_mean / var_sqrt
            self._cache = [X_mean, X_hat, var_sqrt]
        
        return A
    
    
    def backward(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is either (m, n) or (m, n_h, n_w, n_c).
        
        Returns:
            Gradients reshaped as either (m, n) or (m, n_h, n_w, n_c).
        """
        
        m = dA.shape[0]
        axis = tuple(range(dA.ndim-1))
        X_mean, X_hat, var_sqrt = self._cache
        
        self._dgamma = np.sum(dA * X_hat, axis=axis)
        self._dbeta = np.sum(dA, axis=axis)
        
        dx_hat = dA * self._gamma
        
        dvar = np.sum(dx_hat * X_mean, axis=axis) * -0.5 / var_sqrt**3
        dsq = (1 / m) * np.ones(dA.shape) * dvar
        
        dx1 = (dx_hat / var_sqrt) + (2 * X_mean * dsq)
        dmean = -1 * np.sum(dx1, axis=axis)
        dx2 = (1 / m) * np.ones(dA.shape) * dmean
        dx = dx1 + dx2
        
        return dx
    
    
    def backward2(self, dA, **kwargs):
        
        m = dA.shape[0]
        axis = tuple(range(dA.ndim-1))
        X_mean, X_hat, var_sqrt = self._cache
        
        self._dgamma = np.sum(dA * X_hat, axis=axis)
        self._dbeta = np.sum(dA, axis=axis)
        
        dx_hat = dA * self._gamma
        
        dvar = np.sum(dx_hat * X_mean, axis=axis) * -0.5 / var_sqrt**3
        dmean = np.sum(dx_hat / -var_sqrt, axis=axis)
        dmean += dvar * np.mean(-2. * X_mean, axis=axis)
        
        dX = (dx_hat / var_sqrt) + (2 * X_mean * dvar / m) + (dmean / m)
        
        return dX
    
    
    def backward3(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, ?).
        
        Returns:
            Gradients reshaped as (m, n_h, n_w, n_c).
        """
        
        m = dA.shape[0]
        axis = tuple(range(dA.ndim-1))
        
        X_mean, X_hat, var_sqrt = self._cache
        
        self._dgamma = np.sum(dA * X_hat, axis=axis)
        self._dbeta = np.sum(dA, axis=axis)
        
        dx_hat = dA * self._gamma
        dx = m * dx_hat
        dx -= X_hat * np.sum(dx_hat * X_hat, axis=axis)
        dx -= np.sum(dx_hat, axis=axis)
        dx /= var_sqrt
        dx /= m
        
        return dx
    
    
    def paramcount(self, shape):
        """
        Calculates number of trainable params.
        
        Args:
            shape: (int,)
                Expected input shape. The shape must be provided without first
                dimension for number of samples (m).
        
        Returns:
            int
                Number of trainable params.
        """
        
        return np.prod(shape) * 2
    
    
    def parameters(self):
        """Gets all layer parameters."""
        
        return self._gamma, self._beta
    
    
    def gradients(self):
        """Gets all layer gradients."""
        
        return self._dgamma, self._dbeta
    
    
    def _init_params(self, shape):
        """Initializes params."""
        
        self._gamma = np.ones(shape)
        self._beta = np.zeros(shape)
        self._mean = 0.0
        self._variance = 1.0
