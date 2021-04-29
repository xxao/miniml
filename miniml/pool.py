#  Created byMartin.cz

import numpy as np
from . layer import Layer
from . utils import im2col, col2im


class MaxPool(Layer):
    """Represents a max-pooling layer of neural network."""
    
    
    def __init__(self, ksize, stride=None):
        """
        Initializes a new instance of Pool.
        
        Args:
            ksize: int or (int, int)
                Size of the kernel as (h, w) or single integer if squared.
            
            stride: int or (int, int) or None
                Single step kernel shift as single value or (s_h, s_w). If set
                to None, full kernel size is used.
        """
        
        self._ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        self._pad = (0, 0, 0, 0)
        self._stride = stride
        
        if stride is None:
            self._stride = tuple(self._ksize)
        elif isinstance(stride, int):
            self._stride = (stride, stride)
        
        self._X_shape = None
        self._cols = None
        self._max_idx = None
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "MaxPool(%dx%d)" % (self._ksize[0], self._ksize[1])
    
    
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
        c_out = c_in
        
        return h_out, w_out, c_out
    
    
    def clear(self):
        """Clears params and caches."""
        
        self._X_shape = None
        self._cols = None
        self._max_idx = None
    
    
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
        
        self._X_shape = X.shape
        
        # get dimensions
        m, h_in, w_in, c_in = self._X_shape
        h_out, w_out, c_out = self.outshape(X.shape[1:])
        
        # apply pooling
        X = X.transpose(0, 3, 1, 2).reshape(m * c_in, 1, h_in, w_in)
        self._cols = im2col(X, self._ksize, self._pad, self._stride)
        self._max_idxs = np.argmax(self._cols, axis=0)
        A = self._cols[self._max_idxs, range(self._max_idxs.size)]
        A = A.reshape(h_out, w_out, m, c_out)
        A = A.transpose(2, 0, 1, 3)
        
        return A
    
    
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
        m, h_in, w_in, c_in = self._X_shape
        
        # apply reversed pooling
        dX_col = np.zeros_like(self._cols)
        dA_flat = dA.transpose(1, 2, 0, 3).ravel()
        dX_col[self._max_idxs, range(self._max_idxs.size)] = dA_flat
        dX = col2im(dX_col, (m * c_in, 1, h_in, w_in), self._ksize, self._pad, self._stride)
        dX = dX.reshape(m, c_in, h_in, w_in)
        dX = dX.transpose(0, 2, 3, 1)
        
        return dX
