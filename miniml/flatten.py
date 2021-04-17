#  Created byMartin.cz

import numpy as np
from . layer import Layer


class Flatten(Layer):
    """Represents a flattening layer of neural network."""
    
    
    def __init__(self):
        """Initializes a new instance of Flatten."""
        
        self._shape = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "Flatten"
    
    
    def forward(self, X, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, ..., n_C).
        
        Returns:
            Flattened data.
        """
        
        self._shape = X.shape
        return np.ravel(X).reshape(X.shape[0], -1)
    
    
    def backward(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, 1).
        
        Returns:
            Gradients reshaped as (m, ..., n_C).
        """
        
        return dA.reshape(self._shape)
