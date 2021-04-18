#  Created byMartin.cz

import numpy as np
from . enums import *
from . dense import Dense


class Model(object):
    """Represents a neural network model."""
    
    
    def __init__(self):
        """Initializes a new instance of Model."""
        
        self._layers = []
    
    
    def __str__(self):
        """Gets string representation."""
        
        layers = [str(layer) for layer in self._layers]
        return "Model[%s]" % " -> ".join(layers)
    
    
    def __repr__(self):
        """Gets debug representation."""
        
        return self.__str__()
    
    
    def __len__(self):
        """Gets number of layers within the model."""
        
        return len(self._layers)
    
    
    @property
    def layers(self):
        """Gets all model layers."""
        
        return tuple(self._layers)
    
    
    def add(self, nodes, activation=RELU, init_method=HE):
        """
        Creates and adds additional dense layer to the model.
        
        Args:
            nodes: int
                Number of output connections (neurons).
            
            activation: str
                Activation function name such as 'sigmoid', 'relu', 'tanh'
                or 'softmax'.
            
            init_method: str
                W parameter initialization method such as 'plain', 'xavier'
                or 'he'.
        """
        
        layer = Dense(nodes, activation, init_method)
        self._layers.append(layer)
    
    
    def reset(self):
        """Resets params and caches in all layers."""
        
        for layer in self._layers:
            layer.reset()
    
    
    def predict(self, X):
        """
        Performs prediction through model layers.
        
        Args:
            X:
                Input data.
        
        Returns:
            Y_hat:
                Predicted output.
        """
        
        return self.forward(X)
    
    
    def forward(self, A, keep=1):
        """
        Performs forward propagation through all layers.
        
        Args:
            A:
                Input data.
            
            keep: float
                Dropout keep probability (0-1).
        
        Returns:
            Y_hat:
                Activations from the output layer.
        """
        
        # process hidden layers
        for layer in self._layers[:-1]:
            A = layer.forward(A, keep=keep)
        
        # process output layer
        Y_hat = self._layers[-1].forward(A)
        
        return Y_hat
    
    
    def backward(self, dA, lamb=0):
        """
        Performs backward propagation through all layers.
        
        Args:
            dA:
                Gradients of cost.
            
            lamb: float
                Lambda parameter for L2 regularization.
        """
        
        # process output layer
        dA = self._layers[-1].backward(dA, lamb=lamb)
        
        # process hidden layers
        for layer in reversed(self._layers[:-1]):
            dA = layer.backward(dA, lamb=lamb)
    
    
    def update(self, optimizer=GD, **optimizer_params):
        """
        Updates params in all layers by specified optimizer.
        
        Args:
            optimizer: str
                Optimizer name such as 'gd', 'momentum', 'rmsprop',
                'adagrad' or 'adam'.
            
            optimizer_params: {str:any}
                Specific optimizer params.
        """
        
        for layer in reversed(self._layers):
            layer.update(optimizer, **optimizer_params)
