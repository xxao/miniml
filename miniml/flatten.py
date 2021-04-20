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
        
        return (np.prod(shape), )
    
    
    def clear(self):
        """Clears params and caches."""
        
        self._shape = None
    
    
    def forward(self, X, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, n_h, n_w, n_c).
        
        Returns:
            Flattened data into (m, n_h x n_w x n_c).
        """
        
        self._shape = X.shape
        return np.ravel(X).reshape(X.shape[0], -1)
    
    
    def backward(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, ?).
        
        Returns:
            Gradients reshaped as (m, n_h, n_w, n_c).
        """
        
        return dA.reshape(self._shape)
