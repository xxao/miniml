#  Created byMartin.cz

import numpy as np
from . enums import *
from . layer import Layer


class Conv2D(Layer):
    """Represents a 2D convolution layer of neural network."""
    
    
    def __init__(self, depth, ksize, stride, pad=SAME, init_method=PLAIN):
        """
        Initializes a new instance of Conv2D.
        
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
                W initialization method name such as 'plain', 'xavier' or 'he'.
        """
        
        self._depth = int(depth)
        self._ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        self._stride = int(stride)
        self._pad = self._init_padding(pad, *self._ksize)
        self._init_method = init_method
        
        self._X = None
        
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "Conv2D(%dx%dx%d)" % (self._ksize[0], self._ksize[0], self._depth)
    
    
    @property
    def W(self):
        """Gets current weights."""
        
        return self._W
    
    
    @property
    def b(self):
        """Gets current biases."""
        
        return self._b
    
    
    def reset(self):
        """Resets params and caches in all layers."""
        
        self._X = None
        
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
    
    
    def forward(self, X, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, n_h, n_w, n_c).
        
        Returns:
            Calculated activations from this layer.
        """
        
        self._X = X
        
        # get dimensions
        m, h_in, w_in, c_in = X.shape
        f_h, f_w = self._ksize
        p_h, p_w = self._pad
        h_out = int(1 + (h_in - f_h + 2*p_h) / self._stride)
        w_out = int(1 + (w_in - f_w + 2*p_w) / self._stride)
        c_out = self._depth
        
        # init params
        if self._W is None:
            self._init_params(f_h, f_w, c_in, c_out)
        
        # init output
        output = np.zeros((m, h_out, w_out, c_out))
        
        # apply padding
        X_pad = self._zero_pad(X)
        
        # loop over vertical axis
        for h in range(h_out):
            h_start = h * self._stride
            h_end = h_start + f_h
            
            # loop over horizontal axis
            for w in range(w_out):
                w_start = w * self._stride
                w_end = w_start + f_w
                
                conv = X_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] * self._W[np.newaxis, :, :, :]
                output[:, h, w, :] = np.sum(conv, axis = (1, 2, 3))
        
        return output + self._b
    
    
    def backward(self, dA, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, n_h, n_w, n_c).
        
        Returns:
            Gradients from this layer.
        """
        
        # get dimensions
        m, h_in, w_in, c_in = self._X.shape
        m, h_out, w_out, c_out = dA.shape
        f_h, f_w = self._ksize
        p_h, p_w = self._pad
        
        # apply padding
        X_pad = self._zero_pad(self._X)
        
        # init output
        output = np.zeros_like(X_pad)
        self._dW = np.zeros_like(self._W)
        self._db = dA.sum(axis=(0, 1, 2)) / m
        
        # loop over vertical axis
        for h in range(h_out):
            h_start = h * self._stride
            h_end = h_start + f_h
            
            # loop over horizontal axis
            for w in range(w_out):
                w_start = w * self._stride
                w_end = w_start + f_w
                
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._W[np.newaxis, :, :, :, :] *
                    dA[:, h:h+1, w:w+1, np.newaxis, :],
                    axis = 4)
                
                self._dW += np.sum(
                    X_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    dA[:, h:h+1, w:w+1, np.newaxis, :],
                    axis = 0)
        
        self._dW /= m
        output = output[:, p_h:p_h+h_in, p_w:p_w+w_in, :]
        
        return output
    
    
    def _zero_pad(self, x):
        """Apply zero-padding."""
        
        return np.pad(x, ((0, 0), (self._pad[0], self._pad[0]), (self._pad[1], self._pad[1]), (0, 0)),
            mode = 'constant',
            constant_values = (0, 0))
    
    
    def _init_padding(self, pad, f_h, f_w):
        """Initialize padding."""
        
        self._pad = (0, 0)
        
        if isinstance(pad, int):
            return pad, pad
        
        elif pad == SAME:
            return (f_h - 1) // 2, (f_w - 1) // 2
        
        return 0, 0
    
    
    def _init_params(self, f_h, f_w, c_in, c_out):
        """Initializes params."""
        
        # init weights
        if self._init_method == PLAIN:
            self._W = np.random.randn(f_h, f_w, c_in, c_out) * 0.01
        
        elif self._init_method == XAVIER:
            self._W = np.random.randn(f_h, f_w, c_in, c_out) * np.sqrt(1 / c_in)
        
        elif self._init_method == HE:
            self._W = np.random.randn(f_h, f_w, c_in, c_out) * np.sqrt(2 / c_in)
        
        else:
            raise ValueError("Unknown initialization method specified! -> '%s" % self._init_method)
        
        # init bias
        # self._b = np.zeros(c_out)
        self._b = np.random.randn(c_out)
