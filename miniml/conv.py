#  Created byMartin.cz

import numpy as np
from . enums import *
from . activations import *
from . layer import Layer


class Conv2D(Layer):
    """Represents a 2D convolution layer of neural network."""
    
    
    def __init__(self, depth, ksize, stride, pad=VALID, activation=RELU, init_method=PLAIN):
        """
        Initializes a new instance of Conv2D.
        
        Args:
            depth: int
                Number of filters.
            
            ksize: int or (int, int)
                Size of the kernel as (n_h, n_w) or single integer if squared.
            
            stride: int
                Single step kernel shift.
            
            pad: str, int, (int, int) or (int, int, int, int)
                Initial data padding as 'valid' or 'same' mode or direct values
                as (p_h, p_w) or (p_t, p_b, p_l, p_r).
            
            activation: str or None
                Activation function name such as 'sigmoid', 'relu' or 'tanh'.
                If set to None, activation is not applied.
            
            init_method: str
                W initialization method name such as 'plain', 'xavier' or 'he'.
        """
        
        self._depth = int(depth)
        self._ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        self._stride = int(stride)
        self._pad = self._init_padding(pad, *self._ksize)
        self._activation = self._init_activation(activation)
        self._init_method = init_method
        
        self._X = None
        self._A = None
        
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "Conv2D(%dx%dx%d|%s)" % (self._ksize[0], self._ksize[0], self._depth, self._activation)
    
    
    @property
    def W(self):
        """Gets current weights."""
        
        return self._W
    
    
    @property
    def b(self):
        """Gets current biases."""
        
        return self._b
    
    
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
        
        h_in, w_in, c_in = shape
        f_h, f_w = self._ksize
        p_t, p_b, p_l, p_r = self._pad
        h_out = int(1 + (h_in - f_h + p_t + p_b) / self._stride)
        w_out = int(1 + (w_in - f_w + p_l + p_r) / self._stride)
        c_out = self._depth
        
        return h_out, w_out, c_out
    
    
    def clear(self):
        """Clears params and caches."""
        
        self._X = None
        self._A = None
        
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
    
    
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
        
        # clear params and caches
        self.clear()
        
        # get dimensions
        h_in, w_in, c_in = shape
        f_h, f_w = self._ksize
        h_out, w_out, c_out = self.outshape(shape)
        
        # init params
        self._init_params(f_h, f_w, c_in, c_out)
        
        # return output shape
        return h_out, w_out, c_out
    
    
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
        p_t, p_b, p_l, p_r = self._pad
        h_out = int(1 + (h_in - f_h + p_t + p_b) / self._stride)
        w_out = int(1 + (w_in - f_w + p_l + p_r) / self._stride)
        c_out = self._depth
        
        # init params
        if self._W is None:
            self._init_params(f_h, f_w, c_in, c_out)
        
        # init output
        Z = np.zeros((m, h_out, w_out, c_out))
        
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
                Z[:, h, w, :] = np.sum(conv, axis = (1, 2, 3))
        
        # apply biases
        Z = Z + self._b
        
        # apply activation
        self._A = Z
        if self._activation is not None:
            self._A = self._activation.forward(Z)
        
        return self._A
    
    
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
        p_t, p_b, p_l, p_r = self._pad
        
        # apply activation
        dZ = dA
        if self._activation is not None:
            dZ = self._activation.backward(self._A, dA)
        
        # apply padding
        X_pad = self._zero_pad(self._X)
        
        # init output
        dA = np.zeros_like(X_pad)
        self._dW = np.zeros_like(self._W)
        self._db = dZ.sum(axis=(0, 1, 2)) / m
        
        # loop over vertical axis
        for h in range(h_out):
            h_start = h * self._stride
            h_end = h_start + f_h
            
            # loop over horizontal axis
            for w in range(w_out):
                w_start = w * self._stride
                w_end = w_start + f_w
                
                dA[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._W[np.newaxis, :, :, :, :] *
                    dZ[:, h:h+1, w:w+1, np.newaxis, :],
                    axis = 4)
                
                self._dW += np.sum(
                    X_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    dZ[:, h:h+1, w:w+1, np.newaxis, :],
                    axis = 0)
        
        self._dW /= m
        dA = dA[:, p_t:p_t+h_in, p_l:p_l+w_in, :]
        
        return dA
    
    
    def update(self, W, b):
        """
        Updates layer params.
        
        Args:
            W: np.ndarray
                Weights.
            
            b: np.ndarray
                Biases.
        """
        
        self._W = W
        self._b = b
    
    
    def _zero_pad(self, x):
        """Apply zero-padding."""
        
        p_t, p_b, p_l, p_r = self._pad
        
        return np.pad(x, ((0, 0), (p_t, p_b), (p_l, p_r), (0, 0)),
            mode = 'constant',
            constant_values = (0, 0))
    
    
    def _init_activation(self, activation):
        """Initializes activation function."""
        
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
        
        elif activation == TANH:
            return Tanh()
        
        raise ValueError("Unsupported activation function specified! -> '%s" % activation)
    
    
    def _init_padding(self, pad, f_h, f_w):
        """Initialize padding."""
        
        if isinstance(pad, int):
            return pad, pad, pad, pad
        
        if pad == VALID:
            return 0, 0, 0, 0
        
        if pad == SAME:
            p_t = (f_h - 1) // 2
            p_b = p_t
            p_l = (f_w - 1) // 2
            p_r = p_l
            
            if not self._ksize[0] % 2:
                p_b += 1
            
            if not self._ksize[1] % 2:
                p_r += 1
            
            return p_t, p_b, p_l, p_r
        
        if len(pad) == 2:
            return pad[0], pad[0], pad[1], pad[1]
        
        if len(pad) == 4:
            return pad
        
        raise ValueError("Incorrect padding! -> %s" % str(pad))
    
    
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
