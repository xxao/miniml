#  Created byMartin.cz

import numpy as np
from . enums import *
from . activations import *
from . layer import Layer


class Dense(Layer):
    """Represents a fully connected linear layer of neural network."""
    
    OPTIMIZE = True
    
    
    def __init__(self, nodes, activation=RELU, init_method=HE):
        """
        Initializes a new instance of Dense.
        
        Args:
            nodes: int
                Number of output connections (neurons).
            
            activation: str or None
                Activation function name such as 'sigmoid', 'relu', 'tanh' or
                'softmax'. If set to None, activation is not applied.
            
            init_method: str
                W initialization method name such as 'plain', 'xavier' or 'he'.
        """
        
        self._nodes = int(nodes)
        self._init_method = init_method
        self._activation = self._init_activation(activation)
        
        self._X = None
        self._A = None
        
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
        
        self._keep = 1
        self._mask = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "Dense(%d | %s | %s)" % (self._nodes, self._init_method, self._activation)
    
    
    @property
    def W(self):
        """Gets current weights."""
        
        return self._W
    
    
    @property
    def b(self):
        """Gets current biases."""
        
        return self._b
    
    
    @property
    def dW(self):
        """Gets current weights gradients."""
        
        return self._dW
    
    
    @property
    def db(self):
        """Gets current biases gradients."""
        
        return self._db
    
    
    def reset(self):
        """Resets params and caches in all layers."""
        
        self._X = None
        self._A = None
        
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
        
        self._keep = 1
        self._mask = None
    
    
    def forward(self, X, keep=1, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data or activations from previous (left) layer.
                The expected shape is (m, n).
            
            keep: float
                Dropout keep probability (0-1).
        
        Returns:
            Calculated activations from this layer.
        """
        
        self._X = X
        self._keep = keep
        self._mask = None
        
        # init params
        if self._W is None:
            self._init_params(self._X.shape[1], self._nodes)
        
        # forward propagation
        Z = np.dot(self._X, self._W.T) + self._b
        
        # apply activation
        self._A = Z
        if self._activation is not None:
            self._A = self._activation.forward(Z)
        
        # apply dropout
        if self._keep < 1:
            self._mask = np.random.rand(self._A.shape[1], self._A.shape[0]).T
            self._mask = (self._mask < self._keep).astype(int)
            self._A = np.multiply(self._A, self._mask)
            self._A = self._A / self._keep
        
        return self._A
    
    
    def backward(self, dA, lamb=0, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA: np.ndarray
                Gradients from previous (right) layer.
                The expected shape is (m, n).
            
            lamb: float
                Lambda parameter for L2 regularization.
        
        Returns:
            Gradients from this layer.
        """
        
        m = self._X.shape[0]
        
        # apply dropout
        if self._keep < 1:
            dA = np.multiply(dA, self._mask)
            dA = dA / self._keep
        
        # apply activation
        dZ = dA
        if self._activation is not None:
            dZ = self._activation.backward(self._A, dA)
        
        # calc gradients
        self._dW = (1 / m) * np.dot(dZ.T, self._X) + (lamb / m) * self._W
        self._db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        
        dA = np.dot(dZ, self._W)
        
        return dA
    
    
    def update(self, W, b):
        """
        Updates layer params.
        
        Args:
            W: np.ndarray
                Weights.
            
            b: np.ndarray
                Biases.
        """
        
        assert(W.shape == self._W.shape)
        assert(b.shape == self._b.shape)
        
        self._W = W
        self._b = b
    
    
    def _init_activation(self, activation):
        """Initializes activation function."""
        
        if activation is None:
            return None
        
        if isinstance(activation, Activation):
            return activation
        
        if activation == LINEAR:
            return Linear()
        
        if activation == SIGMOID:
            return Sigmoid()
        
        elif activation == RELU:
            return ReLU()
        
        elif activation == TANH:
            return Tanh()
        
        elif activation == SOFTMAX:
            return Softmax()
        
        raise ValueError("Unknown activation function specified! -> '%s" % activation)
    
    
    def _init_params(self, n_in, n_out):
        """Initializes params."""
        
        # init weights
        if self._init_method == PLAIN:
            self._W = np.random.randn(n_out, n_in) * 0.01
        
        elif self._init_method == XAVIER:
            self._W = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)
        
        elif self._init_method == HE:
            self._W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
        
        else:
            raise ValueError("Unknown initialization method specified! -> '%s" % self._init_method)
        
        # init bias
        self._b = np.zeros((1, n_out))
