#  Created byMartin.cz


class Layer(object):
    """Represents a base class for various types of neural network layer."""
    
    
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
        
        Returns:
            Gradients from this layer.
        """
        
        pass
