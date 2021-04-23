#  Created byMartin.cz


class Layer(object):
    """Represents a base class for various types of neural network layers."""
    
    OPTIMIZE = False
    
    
    def __repr__(self):
        """Gets debug representation."""
        
        return self.__str__()
    
    
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
        
        return shape
    
    
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
        
        return 0
    
    
    def clear(self):
        """Clears params and caches."""
        
        pass
    
    
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
        
        # return output shape
        return self.outshape(shape)
    
    
    def forward(self, X, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, ...).
        
        Returns:
            Output data/activations.
        """
        
        return X
    
    
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
        
        pass
