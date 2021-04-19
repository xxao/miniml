#  Created byMartin.cz


class Layer(object):
    """Represents a base class for various types of neural network layers."""
    
    OPTIMIZE = False
    
    
    def __repr__(self):
        """Gets debug representation."""
        
        return self.__str__()
    
    
    def reset(self):
        """Resets params and caches in all layers."""
        
        pass
    
    
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
        
        pass
    
    
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
        
        pass
    
    
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
