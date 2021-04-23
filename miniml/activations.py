#  Created byMartin.cz

import numpy as np
from .enums import *
from . layer import Layer


class Activation(Layer):
    """Represents a baseclass for activation function."""
    
    
    def __str__(self):
        """Gets string representation."""
        
        return self.__class__.__name__
    
    
    @staticmethod
    def create(activation, **kwargs):
        """
        Args:
            activation: str or None
                Activation function name such as 'linear', 'sigmoid', 'relu', 'tanh' or
                'softmax'. If set to None, activation is not applied.
        
        Return:
            miniml.Activation
                Activation layer.
        """
        
        if activation is None:
            return None
        
        if activation is None:
            return None
        
        if isinstance(activation, Activation):
            return activation
        
        if activation == LINEAR:
            return Linear()
        
        if activation == SIGMOID:
            return Sigmoid()
        
        elif activation == RELU:
            return ReLU()
        
        elif activation == LRELU:
            return LeakyReLU()
        
        elif activation == TANH:
            return Tanh()
        
        elif activation == SOFTMAX:
            return Softmax()
        
        raise ValueError("Unknown activation function specified! -> '%s" % activation)


class Linear(Activation):
    """Represents a linear (identity) activation function."""
    
    
    def forward(self, Z, **kwargs):
        """Performs forward propagation through activation function."""
        
        return Z
    
    
    def backward(self, dA, **kwargs):
        """Performs backward propagation through activation function."""
        
        return dA * 1


class Sigmoid(Activation):
    """Represents a sigmoid activation function."""
    
    
    def forward(self, Z, **kwargs):
        """Performs forward propagation through activation function."""
        
        self._A = 1 / (1 + np.exp(-Z))
        return self._A
    
    
    def backward(self, dA, **kwargs):
        """Performs backward propagation through activation function."""
        
        return dA * self._A * (1 - self._A)


class ReLU(Activation):
    """Represents a ReLU activation function."""
    
    
    def forward(self, Z, **kwargs):
        """Performs forward propagation through activation function."""
        
        self._A = np.maximum(Z, 0)
        return self._A
    
    
    def backward(self, dA, **kwargs):
        """Performs backward propagation through activation function."""
        
        dZ = np.array(dA, copy=True)
        dZ[self._A <= 0] = 0
        return dZ


class LeakyReLU(Activation):
    """Represents a LeakyReLU activation function."""
    
    ALPHA = 0.01
    
    def forward(self, Z, **kwargs):
        """Performs forward propagation through activation function."""
        
        self._A = np.array(Z, copy=True)
        self._A[Z < 0] = Z[Z < 0] * self.ALPHA
        return self._A
    
    
    def backward(self, dA, **kwargs):
        """Performs backward propagation through activation function."""
        
        dZ = np.ones_like(self._A)
        dZ[self._A < 0] *= self.ALPHA
        return dA * dZ


class Tanh(Activation):
    """Represents a Tanh activation function."""
    
    
    def forward(self, Z, **kwargs):
        """Performs forward propagation through activation function."""
        
        self._A = np.tanh(Z)
        return self._A
    
    
    def backward(self, dA, **kwargs):
        """Performs backward propagation through activation function."""
        
        return dA * (1 - np.power(self._A, 2))


class Softmax(Activation):
    """Represents a Softmax activation function."""
    
    
    def forward(self, Z, **kwargs):
        """Performs forward propagation through activation function."""
        
        eZ = np.exp(Z - np.max(Z))
        self._A = eZ / eZ.sum(axis=1, keepdims=True)
        return self._A
    
    
    def backward(self, dA, **kwargs):
        """Performs backward propagation through activation function."""
        
        m, C = self._A.shape
        dZ = np.zeros((m, C))
        ones = np.ones((1, C))
        ident = np.identity(C)
        
        for j in range(m):
            
            a = self._A[j].reshape(C, 1)
            da = dA[j].reshape(C, 1)
            
            t = np.matmul(a, ones)
            mat = t * (ident - t.T)
            dz = np.matmul(mat, da)
            dZ[j] = dz.T[0]
        
        return dZ
