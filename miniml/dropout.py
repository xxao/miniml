#  Created byMartin.cz

import numpy as np
from . layer import Layer


class Dropout(Layer):
    """Represents a dropout layer of neural network."""
    
    
    def __init__(self, keep):
        """
        Initializes a new instance of Dropout.
        
        Args:
            keep: float
                Keep probability as %/100.
        """
        
        self._keep = float(keep)
        self._mask = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "Dropout(%.2f)" % self._keep
    
    
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
        self._mask = None
        
        # return output shape
        return self.outshape(shape)
    
    
    def forward(self, X, training, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, ...).
            
            training: bool
                If set to True, the input data/activations are considered as
                training set.
        
        Returns:
            Output data/activations.
        """
        
        if not training or self._keep == 1:
            return X
        
        self._mask = np.random.rand(X.shape[1], X.shape[0]).T
        self._mask = (self._mask < self._keep).astype(int)
        A = np.multiply(X, self._mask)
        A = A / self._keep
        
        return A
    
    
    def backward(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, ?).
        
        Returns:
            Gradients from this layer.
        """
        
        if self._keep == 1:
            return dA
        
        dX = np.multiply(dA, self._mask)
        dX = dX / self._keep
        
        return dX
