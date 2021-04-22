#  Created byMartin.cz

import numpy as np


class Activation(object):
    """Represents a baseclass for activation function."""
    
    
    def __str__(self):
        """Gets string representation."""
        
        return self.__class__.__name__
    
    
    def __repr__(self):
        """Gets debug representation."""
        
        return self.__str__()
    
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        raise NotImplementedError
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        raise NotImplementedError


class Linear(Activation):
    """Represents a linear (identity) activation function."""
    
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        return Z
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        return dA * 1


class Sigmoid(Activation):
    """Represents a sigmoid activation function."""
    
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        return 1 / (1 + np.exp(-Z))
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        return dA * A * (1 - A)


class ReLU(Activation):
    """Represents a ReLU activation function."""
    
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        return np.maximum(Z, 0)
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        dZ = np.array(dA, copy=True)
        dZ[A <= 0] = 0
        return dZ


class LeakyReLU(Activation):
    """Represents a LeakyReLU activation function."""
    
    ALPHA = 0.01
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        A = np.array(Z, copy=True)
        A[A < 0] = A[A < 0] * self.ALPHA
        return A
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        dZ = np.ones_like(A)
        dZ[A < 0] *= self.ALPHA
        return dA * dZ


class Tanh(Activation):
    """Represents a Tanh activation function."""
    
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        return np.tanh(Z)
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        return dA * (1 - np.power(A, 2))


class Softmax(Activation):
    """Represents a Softmax activation function."""
    
    
    def forward(self, Z):
        """Performs forward propagation through activation function."""
        
        eZ = np.exp(Z - np.max(Z))
        return eZ / eZ.sum(axis=1, keepdims=True)
    
    
    def backward(self, A, dA):
        """Performs backward propagation through activation function."""
        
        m, C = A.shape
        dZ = np.zeros((m, C))
        ones = np.ones((1, C))
        ident = np.identity(C)
        
        for j in range(m):
            
            a = A[j].reshape(C, 1)
            da = dA[j].reshape(C, 1)
            
            t = np.matmul(a, ones)
            mat = t * (ident - t.T)
            dz = np.matmul(mat, da)
            dZ[j] = dz.T[0]
        
        return dZ
