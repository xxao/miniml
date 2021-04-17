#  Created byMartin.cz

import numpy as np
from . enums import *
from . activations import *
from . layer import Layer


class Dense(Layer):
    """Represents a fully connected linear layer of neural network."""
    
    
    def __init__(self, n_in, n_out, activation=RELU, w_init=HE):
        """
        Initializes a new instance of Dense.
        
        Args:
            n_in: int
                Number of input connections (features).
            
            n_out: int
                Number of output connections (neurons).
            
            activation: str
                Activation function name such as 'sigmoid', 'relu', 'tanh' or 'softmax'.
            
            w_init: str
                W initialization method name such as 'plain', 'xavier' or 'he'.
        """
        
        self._n_in = int(n_in)
        self._n_out = int(n_out)
        self._w_init = w_init
        
        # set activation function
        self._activation = self._init_activation(activation)
        
        # initializes params and caches
        self.initialize()
    
    
    def __str__(self):
        """Gets string representation."""
        
        return "Linear(%d, %d, %s) | %s" % (self._n_in, self._n_out, self._w_init, self._activation)
    
    
    def __len__(self):
        """Gets number of neurons within the layer."""
        
        return self._n_out
    
    
    @property
    def W(self):
        """Gets current weights."""
        
        return self._W
    
    
    @property
    def b(self):
        """Gets current biases."""
        
        return self._b
    
    
    def initialize(self, optimizer=GD, **kwargs):
        """
        Resets all internal caches and re-initializes params.
        
        Args:
            optimizer: str
                Optimizer name.
        """
        
        # main input and output
        self._X = None
        self._A = None
        
        # params to learn
        self._W = None
        self._b = None
        
        # params gradients
        self._dW = None
        self._db = None
        
        # backdrop
        self._keep = 1
        self._mask = None
        
        # momentum and adam
        self._t = 0
        self._vdW = None
        self._vdb = None
        self._sdW = None
        self._sdb = None
        
        # init params
        self._init_params(optimizer)
    
    
    def forward(self, X, keep=1, **kwargs):
        """
        Performs forward propagation using activations from previous layer.
        
        Args:
            X:
                Input data or activations from previous (left) layer.
            
            keep: float
                Dropout keep probability (0-1).
        
        Returns:
            Calculated activations from this layer.
        """
        
        self._X = X
        self._keep = keep
        self._mask = None
        
        # forward propagation
        Z = np.dot(self._W, self._X) + self._b
        self._A = self._activation.forward(Z)
        
        # apply dropout
        if self._keep < 1:
            self._mask = np.random.rand(self._A.shape[0], self._A.shape[1])
            self._mask = (self._mask < self._keep).astype(int)
            self._A = np.multiply(self._A, self._mask)
            self._A = self._A / self._keep
        
        return self._A
    
    
    def backward(self, dA, lamb=0, **kwargs):
        """
        Performs backward propagation using upstream gradients.
        
        Args:
            dA:
                Gradients from previous (right) layer.
            
            lamb: float
                Lambda parameter for L2 regularization.
        
        Returns:
            Gradients from this layer.
        """
        
        m = self._X.shape[1]
        
        # apply dropout
        if self._keep < 1:
            dA = np.multiply(dA, self._mask)
            dA = dA / self._keep
        
        # calc gradients
        dZ = self._activation.backward(self._A, dA)
        self._dW = (1 / m) * np.dot(dZ, self._X.T) + (lamb / m) * self._W
        self._db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        dA = np.dot(self._W.T, dZ)
        
        return dA
    
    
    def update(self, optimizer=GD, **optimizer_params):
        """
        Updates params by specified optimizer.
        
        Args:
            optimizer: str
                Optimizer name.
            
            optimizer_params: {str:any}
                Specific optimizer params.
        """
        
        # update by gradient descent
        if optimizer == GD:
            self.update_gd(**optimizer_params)
        
        # update by gradient descent with Momentum.
        elif optimizer == MOMENTUM:
            self.update_momentum(**optimizer_params)
        
        # update by gradient descent with RMSprop.
        elif optimizer == RMSPROP:
            self.update_rmsprop(**optimizer_params)
        
        # update by gradient descent with Adam.
        elif optimizer == ADAM:
            self.update_adam(**optimizer_params)
        
        # update by gradient descent with Adagrad.
        elif optimizer == ADAGRAD:
            self.update_adagrad(**optimizer_params)
        
        # unknown optimizer
        else:
            raise ValueError("Unknown optimizer specified! -> '%s" % optimizer)
    
    
    def update_gd(self, rate=0.1):
        """
        Updates params by gradient descent.
        
        Args:
            rate: float
                Learning rate.
        """
        
        self._W = self._W - rate * self._dW
        self._b = self._b - rate * self._db
    
    
    def update_momentum(self, rate=0.1, beta=0.9):
        """
        Updates params by gradient descent with Momentum.
        
        Args:
            rate: float
                Learning rate.
            
            beta:
                Momentum parameter.
        """
        
        self._vdW = beta * self._vdW + (1 - beta) * self._dW
        self._vdb = beta * self._vdb + (1 - beta) * self._db
        
        self._W = self._W - rate * self._vdW
        self._b = self._b - rate * self._vdb
    
    
    def update_rmsprop(self, rate=0.1, beta=0.9, epsilon=1e-8):
        """
        Updates params by gradient descent with RMSprop.
        
        Args:
            rate: float
                Learning rate.
            
            beta: float
                Momentum parameter.
            
            epsilon: float
                Zero division corrector.
        """
        
        self._sdW = beta * self._sdW + (1 - beta) * self._dW**2
        self._sdb = beta * self._sdb + (1 - beta) * self._db**2
        
        self._W = self._W - rate * self._dW / np.sqrt(self._sdW + epsilon)
        self._b = self._b - rate * self._db / np.sqrt(self._sdb + epsilon)
    
    
    def update_adam(self, rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Updates params by gradient descent with Adam.
        
        Args:
            rate: float
                Learning rate.
            
            beta1: float
                First momentum parameter.
            
            beta2: float
                Second momentum parameter.
            
            epsilon: float
                Zero division corrector.
        """
        
        self._t += 1
        
        self._vdW = beta1 * self._vdW + (1 - beta1) * self._dW
        self._vdb = beta1 * self._vdb + (1 - beta1) * self._db
        
        v_corr_dW = self._vdW / (1 - beta1**self._t)
        v_corr_db = self._vdb / (1 - beta1**self._t)
        
        self._sdW = beta2 * self._sdW + (1 - beta2) * self._dW**2
        self._sdb = beta2 * self._sdb + (1 - beta2) * self._db**2
        
        s_corr_dW = self._sdW / (1 - beta2**self._t)
        s_corr_db = self._sdb / (1 - beta2**self._t)
        
        self._W = self._W - rate * v_corr_dW / (s_corr_dW**0.5 + epsilon)
        self._b = self._b - rate * v_corr_db / (s_corr_db**0.5 + epsilon)
    
    
    def update_adagrad(self, rate=0.1, epsilon=1e-8):
        """
        Updates params by gradient descent with Adagrad.
        
        Args:
            rate: float
                Learning rate.
            
            epsilon: float
                Zero division corrector.
        """
        
        self._sdW = self._sdW + self._dW**2
        self._sdb = self._sdb + self._db**2
        
        self._W = self._W - rate * self._dW / np.sqrt(self._sdW + epsilon)
        self._b = self._b - rate * self._db / np.sqrt(self._sdb + epsilon)
    
    
    def _init_activation(self, activation):
        """Initializes activation function."""
        
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
        
        elif activation == SOFTMAX:
            return Softmax()
        
        raise ValueError("Unknown activation function specified! -> '%s" % activation)
    
    
    def _init_params(self, optimizer):
        """Initializes params."""
        
        # init weights
        if self._w_init == PLAIN:
            self._W = np.random.randn(self._n_out, self._n_in) * 0.01
        
        elif self._w_init == XAVIER:
            self._W = np.random.randn(self._n_out, self._n_in) * np.sqrt(1/self._n_in)
        
        elif self._w_init == HE:
            self._W = np.random.randn(self._n_out, self._n_in) * np.sqrt(2/self._n_in)
        
        else:
            raise ValueError("Unknown initialization method specified! -> '%s" % self._w_init)
        
        # init bias
        self._b = np.zeros((self._n_out, 1))
        
        # init squared
        if optimizer in (RMSPROP, ADAM, ADAGRAD):
            self._sdW = np.zeros((self._n_out, self._n_in))
            self._sdb = np.zeros((self._n_out, 1))
        
        # init velocity
        if optimizer in (MOMENTUM, ADAM):
            self._vdW = np.zeros((self._n_out, self._n_in))
            self._vdb = np.zeros((self._n_out, 1))
