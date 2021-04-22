#  Created byMartin.cz

import numpy as np
from . enums import *
from . layer import Layer
from . conv import Conv2D
from . dense import Dense
from . dropout import Dropout
from . flatten import Flatten
from . pool import MaxPool


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
    
    
    def summary(self, shape):
        """
        Prints model summary.
        
        Args:
            shape: (int,)
                Expected input shape. The shape must be provided without first
                dimension for number of samples (m).
        """
        
        length = 81
        
        print()
        print("=" * length)
        print(f" {'Layer':<30} | {'Output':<25} | {'Params':>18}")
        print("-" * length)
        
        total_params = 0
        for layer in self._layers:
            
            params = layer.params(shape)
            total_params += params
            
            shape = layer.outshape(shape)
            full_shape = (None, *shape)
            
            print(f" {str(layer):<30} | {str(full_shape):<25} | {params:>18,}")
        
        print("-" * length)
        print(f"{'Total':>59} | {total_params:>18,}")
        print("=" * length)
    
    
    def clear(self):
        """Resets params and caches in all layers."""
        
        for layer in self._layers:
            layer.clear()
    
    
    def initialize(self, shape):
        """
        Clears caches and re-initializes params in all layers.
        
        Args:
            shape: (int,)
                Expected input shape. The shape must be provided without first
                dimension for number of samples (m).
        
        Returns:
            (int,)
                Output shape. The shape is provided without first dimension for
                number of samples (m).
        """
        
        for layer in self._layers:
            shape = layer.initialize(shape)
    
    
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
    
    
    def conv2d(self, depth, ksize, stride, pad=VALID, activation=RELU, init_method=HE):
        """
        Appends new 2D convolution layer.
        
        Args:
            depth: int
                Number of filters.
            
            ksize: int or (int, int)
                Size of the kernel as (n_h, n_w) or single integer if squared.
            
            stride: int or (int, int)
                Single step kernel shift as single value or (s_h, s_w).
            
            pad: str, int, (int, int) or (int, int, int, int)
                Initial data padding as 'valid' or 'same' mode or direct values
                as (p_h, p_w) or (p_t, p_b, p_l, p_r).
            
            activation: str or None
                Activation function name such as 'sigmoid', 'relu' or 'tanh'.
                If set to None, activation is not applied.
            
            init_method: str
                W parameter initialization method such as 'plain', 'xavier'
                or 'he'.
        """
        
        self.add(Conv2D(depth, ksize, stride, pad, activation, init_method))
    
    
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
    
    
    def maxpool(self, ksize, stride):
        """
        Appends new max-pooling layer.
        
        Args:
            ksize: int or (int, int)
                Size of the kernel as (h, w) or single integer if squared.
            
            stride: int or (int, int)
                Single step kernel shift as single value or (s_h, s_w).
        """
        
        self.add(MaxPool(ksize, stride))
