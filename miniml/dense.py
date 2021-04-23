#  Created byMartin.cz

import numpy as np
from . enums import *
from . activations import Activation
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
        self._activation = Activation.create(activation)
        self._init_method = init_method
        
        self._X = None
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        activation = ""
        if self._activation is not None:
            activation = "|%s" % self._activation
        
        return "Dense(%d%s)" % (self._nodes, activation)
    
    
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
    
    
    def outshape(self, shape):
        """
        Calculates output shape.
        
        Args:
            shape: (int,)
                Expected input shape. The shape must be provided without first
                dimension for number of samples (m).
        
        Returns:
            (int,)
                Output shape. The shape is provided without first dimension for
                number of samples (m).
        """
        
        return (self._nodes, )
    
    
    def params(self, shape):
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
        
        # get dimensions
        n_in = shape[0]
        n_out = self._nodes
        
        # count params
        w = n_in * n_out
        b = n_out
        
        return w + b
    
    
    def clear(self):
        """Clears params and caches."""
        
        self._X = None
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
    
    
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
        self.clear()
        
        # init params
        self._init_params(shape[0], self._nodes)
        
        # return output shape
        return self.outshape(shape)
    
    
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
        
        # init params
        if self._W is None:
            self._init_params(self._X.shape[1], self._nodes)
        
        # forward propagation
        Z = np.dot(self._X, self._W.T) + self._b
        
        # apply activation
        if self._activation is not None:
            return self._activation.forward(Z)
        
        return Z
    
    
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
        
        # apply activation
        dZ = dA
        if self._activation is not None:
            dZ = self._activation.backward(dA)
        
        # calc gradients
        self._dW = (1 / m) * np.dot(dZ.T, self._X) + (lamb / m) * self._W
        self._db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        
        dX = np.dot(dZ, self._W)
        
        return dX
    
    
    def update(self, W, b):
        """
        Updates layer params.
        
        Args:
            W: np.ndarray
                Weights.
            
            b: np.ndarray
                Biases.
        """
        
        self._W = W
        self._b = b
    
    
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
