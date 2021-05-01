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
        self._var_sqrt = None
        self._X_mean = None
        self._X_hat = None
    
    
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
        self._var_sqrt = None
        self._X_mean = None
        self._X_hat = None
        
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
        
        # init params
        if self._gamma is None:
            self._init_params(X.shape[1:])
        
        # get mean and variance
        mean = self._mean
        variance = self._variance
        
        # use current batch
        if training:
            axis = tuple(range(X.ndim-1))
            mean = np.mean(X, axis=axis)
            variance = np.mean((X - mean)**2, axis=axis)
        
        # normalize
        var_sqrt = np.sqrt(variance + self._epsilon)
        X_mean = (X - mean)
        X_hat = X_mean / var_sqrt
        
        # scale and shift
        A = self._gamma * X_hat + self._beta
        
        # update running mean and variance and fill cache
        if training:
            self._mean = self._mean * self._momentum + (1 - self._momentum) * mean
            self._variance = self._variance * self._momentum + (1 - self._momentum) * variance
            self._var_sqrt = var_sqrt
            self._X_mean = X_mean
            self._X_hat = X_hat
        
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
        
        self._dgamma = np.sum(dA * self._X_hat, axis=axis)
        self._dbeta = np.sum(dA, axis=axis)
        
        dx_hat = dA * self._gamma
        dx = m * dx_hat
        dx -= self._X_hat * np.sum(dx_hat * self._X_hat, axis=axis)
        dx -= np.sum(dx_hat, axis=axis)
        dx /= self._var_sqrt
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
