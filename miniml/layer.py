#  Created byMartin.cz


class Layer(object):
    """Represents a base class for various types of neural network layers."""
    
    
    def __str__(self):
        """Gets string representation."""
        
        return self.__class__.__name__
    
    
    def __repr__(self):
        """Gets debug representation."""
        
        return self.__str__()
    
    
    @property
    def trainable(self):
        """Returns True if layer has parameters to train."""
        
        return len(self.parameters()) > 0
    
    
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
    
    
    def update(self, *params):
        """
        Updates layer params.
        
        Args:
            *params: (np.ndarray,)
                Params to update.
        """
        
        pass
    
    
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
        
        return 0
    
    
    def parameters(self):
        """Gets all layer parameters."""
        
        return []
    
    
    def gradients(self):
        """Gets all layer gradients."""
        
        return []
    
    
    def loss(self):
        """
        Calculates regularization loss.
        
        Returns:
            float
                Regularization loss.
        """
        
        return 0
