#  Created byMartin.cz

import numpy as np
from . enums import *
from . layer import Layer


class Pool(Layer):
    """Represents a pooling layer of neural network."""
    
    
    def __init__(self, size, stride, mode='max'):
        """
        Initializes a new instance of Pool.
        
        Args:
            size: int or (int, int)
                Size of the kernel as (h, w) or single integer fi square.
            
            stride: int
                Single step kernel shift.
            
            mode: str
                Pooling modes such as 'max' or 'avg'.
        """
        
        self._mode = mode
        self._size = (size, size) if isinstance(size, int) else size
        self._stride = int(stride)
        
        self._X = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "%sPool(%dx%d)" % (self._mode.title(), self._size[0], self._size[1])
    
    
    def forward(self, X, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, n_H, n_W, n_C).
        
        Returns:
            Calculated activations from this layer.
        """
        
        self._X = X
        
        # get dimensions
        m, n_H, n_W, n_C = X.shape
        f_h, f_w = self._size
        n_H = int(1 + (n_H - f_h) / self._stride)
        n_W = int(1 + (n_W - f_w) / self._stride)
        
        # init output
        output = np.zeros((m, n_H, n_W, n_C))
        
        # loop over vertical axis
        for h in range(n_H):
            h_start = h * self._stride
            h_end = h_start + f_h
            
            # loop over horizontal axis
            for w in range(n_W):
                w_start = w * self._stride
                w_end = w_start + f_w
                
                # get slice
                a_slice = X[:, h_start:h_end, w_start:w_end, :]
                
                # max pooling
                if self._mode == MAX:
                    output[:, h, w, :] = np.max(a_slice, axis=(1, 2))
                
                # average pooling
                elif self._mode == AVG:
                    output[:, h, w, :] = np.mean(a_slice, axis=(1, 2))
        
        return output
    
    
    def backward(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, n_H, n_W, n_C).
        
        Returns:
            Gradients from this layer.
        """
        
        # get dimensions
        f_h, f_w = self._size
        m, n_H, n_W, n_C = dA.shape
        
        # init output
        output = np.zeros(self._X.shape)
        
        # loop over samples
        for i in range(m):
            
            # loop over vertical axis
            for h in range(n_H):
                h_start = h * self._stride
                h_end = h_start + f_h
                
                # loop over horizontal axis
                for w in range(n_W):
                    w_start = w * self._stride
                    w_end = w_start + f_w
                    
                    # loop over channels
                    for c in range(n_C):
                        da = dA[i, h, w, c]
                        
                        # max pooling
                        if self._mode == MAX:
                            a_slice = self._X[i, h_start: h_end, w_start: w_end, c]
                            mask = (a_slice == np.max(a_slice))
                            output[i, h_start: h_end, w_start: w_end, c] += mask * da
                        
                        # average pooling
                        elif self._mode == AVG:
                            average = da / (f_h * f_w)
                            output[i, h_start: h_end, w_start: w_end, c] += np.ones((f_h, f_w)) * average
        
        return output


class MaxPool(Pool):
    """Represents a maximum pooling layer of neural network."""
    
    
    def __init__(self, size, stride):
        """
        Initializes a new instance of MaxPool.
        
        Args:
            size: int or (int, int)
                Size of the kernel as (h, w) or single integer fi square.
            
            stride: int
                Single step kernel shift.
        """
        
        super().__init__(size, stride, mode=MAX)


class AvgPool(Pool):
    """Represents a average pooling layer of neural network."""
    
    
    def __init__(self, size, stride):
        """
        Initializes a new instance of AvgPool.
        
        Args:
            size: int or (int, int)
                Size of the kernel as (h, w) or single integer fi square.
            
            stride: int
                Single step kernel shift.
        """
        
        super().__init__(size, stride, mode=AVG)
