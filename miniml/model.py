#  Created byMartin.cz

import numpy as np
from . enums import *
from . layer import Layer
from . conv import Conv2D
from . dense import Dense
from . dropout import Dropout
from . flatten import Flatten
from . pool import Pool


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
    
    
    @property
    def layers(self):
        """Gets all model layers."""
        
        return tuple(self._layers)
    
    
    def reset(self):
        """Resets params and caches in all layers."""
        
        for layer in self._layers:
            layer.reset()
    
    
    def predict(self, A):
        """
        Performs prediction through model layers.
        
        Args:
            A: np.ndarray
                Input data of shape (m,...).
        
        Returns:
            Y_hat:
                Predicted output with shape (m,?).
        """
        
        for layer in self._layers:
            if not isinstance(layer, Dropout):
                A = layer.forward(A)
        
        return A
    
    
    def forward(self, A):
        """
        Performs forward propagation through all layers.
        
        Args:
            A: np.ndarray
                Input data of shape (m,...).
        
        Returns:
            Y_hat:
                Activations from the output layer with shape (m,?).
        """
        
        for layer in self._layers:
            A = layer.forward(A)
        
        return A
    
    
    def backward(self, dA, lamb=0):
        """
        Performs backward propagation through all layers.
        
        Args:
            dA: np.ndarray
                Gradients of cost with shape (m, ?).
            
            lamb: float
                Lambda parameter for L2 regularization.
        """
        
        for layer in reversed(self._layers):
            dA = layer.backward(dA, lamb=lamb)
    
    
    def add(self, layer):
        """
        Appends new layer into model.
        
        Args:
            layer: miniml.Layer
                Layer to be added.
        """
        
        if not isinstance(layer, Layer):
            raise TypeError("Layers must be of type miniml.Layer! -> '%s'" % type(layer))
        
        self._layers.append(layer)
    
    
    def conv2d(self, depth, ksize, stride, pad=VALID, init_method=HE):
        """
        Appends new 2D convolution layer.
        
        Args:
            depth: int
                Number of filters.
            
            ksize: int or (int, int)
                Size of the kernel as (n_h, n_w) or single integer if squared.
            
            stride: int
                Single step kernel shift.
            
            pad: int, (int, int) or str
                Initial data padding as a specific number or mode such as
                'valid' or 'same'.
            
            init_method: str
                W parameter initialization method such as 'plain', 'xavier'
                or 'he'.
        """
        
        self.add(Conv2D(depth, ksize, stride, pad, init_method))
    
    
    def dense(self, nodes, activation=RELU, init_method=HE):
        """
        Appends new fully-connected (dense) layer.
        
        Args:
            nodes: int
                Number of output connections (neurons).
            
            activation: str
                Activation function name such as 'sigmoid', 'relu', 'tanh' or
                'softmax'. If set to None, activation is not applied.
            
            init_method: str
                W parameter initialization method such as 'plain', 'xavier'
                or 'he'.
        """
        
        self.add(Dense(nodes, activation, init_method))
    
    
    def dropout(self, keep):
        """
        Appends new dropout layer.
        
        Args:
            keep: float
                Keep probability as %/100.
        """
        
        self.add(Dropout(keep))
    
    
    def flatten(self):
        """Appends new flattening layer."""
        
        self.add(Flatten())
    
    
    def pool(self, ksize, stride, mode=MAX):
        """
        Appends new pooling layer.
        
        Args:
            ksize: int or (int, int)
                Size of the kernel as (h, w) or single integer if squared.
            
            stride: int
                Single step kernel shift.
            
            mode: str
                Pooling modes such as 'max' or 'avg'.
        """
        
        self.add(Pool(ksize, stride, mode))
