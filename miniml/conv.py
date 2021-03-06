#  Created byMartin.cz

import numpy as np
from . enums import *
from . activations import Activation
from . layer import Layer
from . utils import *


class Conv2D(Layer):
    """Represents a 2D convolution layer of neural network."""
    
    
    def __init__(self, depth, ksize, stride=1, pad=VALID, activation=RELU, init_method=PLAIN):
        """
        Initializes a new instance of Conv2D.
        
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
                W initialization method name such as 'plain', 'xavier' or 'he'.
        """
        
        self._depth = int(depth)
        self._ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        self._stride = (stride, stride) if isinstance(stride, int) else stride
        self._pad = self._init_padding(pad, *self._ksize)
        self._activation = Activation.create(activation)
        self._init_method = init_method
        
        self._X = None
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
        self._cols = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        activation = ""
        if self._activation is not None:
            activation = "|%s" % self._activation
        
        return "Conv2D(%dx%dx%d%s)" % (self._ksize[0], self._ksize[0], self._depth, activation)
    
    
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
        self._X = None
        self._W = None
        self._b = None
        self._dW = None
        self._db = None
        self._cols = None
        
        # get dimensions
        h_in, w_in, c_in = shape
        f_h, f_w = self._ksize
        h_out, w_out, c_out = self.outshape(shape)
        
        # init params
        self._init_params(f_h, f_w, c_in, c_out)
        
        # return output shape
        return h_out, w_out, c_out
    
    
    def forward(self, X, training=None, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X: np.ndarray
                Input data/activations from previous (left) layer.
                The expected shape is (m, n_h, n_w, n_c).
            
            training: bool
                If set to True, the input data/activations are considered as
                training set.
        
        Returns:
            Calculated activations from this layer.
        """
        
        self._X = X
        
        # get dimensions
        m, h_in, w_in, c_in = X.shape
        h_out, w_out, c_out = self.outshape(X.shape[1:])
        f_h, f_w = self._ksize
        
        # init params
        if self._W is None:
            self._init_params(f_h, f_w, c_in, c_out)
        
        # apply convolution
        X = X.transpose(0, 3, 1, 2)
        self._cols = im2col(X, self._ksize, self._pad, self._stride)
        W = self._W.transpose(3, 2, 0, 1)
        Z = W.reshape((c_out, -1)).dot(self._cols)
        Z = Z.reshape(c_out, h_out, w_out, m)
        Z = Z.transpose(3, 1, 2, 0) + self._b
        
        # apply activation
        if self._activation is not None:
            return self._activation.forward(Z)
        
        return Z
    
    
    def backward(self, dA, lamb=0, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
                The expected shape is (m, n_h, n_w, n_c).
            
            lamb: float
                Lambda parameter for L2 regularization.
        
        Returns:
            Gradients from this layer.
        """
        
        # get dimensions
        m, h_out, w_out, c_out = dA.shape
        
        # apply activation
        dZ = dA
        if self._activation is not None:
            dZ = self._activation.backward(dA)
        
        # calc gradients
        dZ = dZ.transpose(3, 1, 2, 0).reshape(c_out, -1)
        W = np.transpose(self._W, (3, 2, 0, 1))
        dW = dZ.dot(self._cols.T).reshape(W.shape)
        
        self._dW = (1 / m) * np.transpose(dW, (2, 3, 1, 0)) + (lamb / m) * self._W
        self._db = (1 / m) * dZ.sum(axis=(0, 1, 2))
        
        # apply reversed convolution
        cols = W.reshape(c_out, -1).T.dot(dZ)
        dX = col2im(cols, np.moveaxis(self._X, -1, 1).shape, self._ksize, self._pad, self._stride)
        dX = np.transpose(dX, (0, 2, 3, 1))
        
        return dX
    
    
    def update(self, *params):
        """
        Updates layer params.
        
        Args:
            *params: (np.ndarray,)
                Params to update as W and b.
        """
        
        self._W = params[0]
        self._b = params[1]
    
    
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
        s_h, s_w = self._stride
        h_out = int(1 + (h_in - f_h + p_t + p_b) / s_h)
        w_out = int(1 + (w_in - f_w + p_l + p_r) / s_w)
        c_out = self._depth
        
        return h_out, w_out, c_out
    
    
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
        
        # get dimensions
        h_in, w_in, c_in = shape
        f_h, f_w = self._ksize
        h_out, w_out, c_out = self.outshape(shape)
        
        # count params
        w = f_h * f_w * c_in * c_out
        b = c_out
        
        return w + b
    
    
    def parameters(self):
        """Gets all layer parameters."""
        
        return self._W, self._b
    
    
    def gradients(self):
        """Gets all layer gradients."""
        
        return self._dW, self._db
    
    
    def loss(self):
        """
        Calculates regularization loss.
        
        Returns:
            float
                Regularization loss.
        """
        
        return np.sum(np.square(self._W))
    
    
    def _zero_pad(self, x):
        """Apply zero-padding."""
        
        p_t, p_b, p_l, p_r = self._pad
        
        return np.pad(x, ((0, 0), (p_t, p_b), (p_l, p_r), (0, 0)),
            mode = 'constant',
            constant_values = (0, 0))
    
    
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
