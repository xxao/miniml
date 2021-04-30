#  Created byMartin.cz

import numpy as np
from . layer import Layer


class Flatten(Layer):
    """Represents a flattening layer of neural network."""
    
    
    def __init__(self):
        """Initializes a new instance of Flatten."""
        
        self._shape = None
    
    
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
        self._shape = None
        
        # return output shape
        return self.outshape(shape)
    
    
    def forward(self, X, training=None, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, n_h, n_w, n_c).
            
            training: bool
                If set to True, the input data/activations are considered as
                training set.
        
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
