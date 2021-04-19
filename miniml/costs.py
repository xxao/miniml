#  Created byMartin.cz

import numpy as np


def mean_squared_error(Y, Y_hat):
    """
    Calculates cost and its gradient using Mean Squared Error.
    
    (1/2m)*sum(Y - Y_hat)^.2
    
    Args:
        Y:
            Data labels.
        
        Y_hat:
            Predictions/activations from the output layer.
    
    Returns:
        cost:
            Mean Squared Error Cost result.
        
        dY_hat:
            Gradient.
    """
    
    m = Y.shape[0]
    
    # calc cost
    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)
    
    # calc gradient
    dY_hat = - (1 / m) * (Y - Y_hat)
    
    return cost, dY_hat


def binary_cross_entropy(Y, Y_hat, epsilon=1e-8):
    """
    Calculates cost and its gradient using Binary Cross Entropy.
    
    (1/m) * np.sum(-np.log(Y_hat) * Y - np.log(1-Y_hat) * (1-Y))
    
    Args:
        Y:
            Data labels.
        
        Y_hat:
            Predictions/activations from the output layer.
        
        epsilon: float
            Zero corrector.
    
    Returns:
        cost:
            Binary Cross-Entropy Cost result.
        
        dY_hat:
            Gradient.
    """
    
    m = Y.shape[0]
    
    # make data safe
    Y_hat = np.clip(Y_hat, a_min=epsilon, a_max=(1 - epsilon))
    
    # calc cost
    cost = (1 / m) * np.nansum(-np.log(Y_hat) * Y - np.log(1 - Y_hat) * (1 - Y))
    cost = np.squeeze(cost)
    
    # calc gradient
    dY_hat = -(Y / Y_hat) + (1 - Y) / (1 - Y_hat)
    
    return cost, dY_hat


def cross_entropy(Y, Y_hat, epsilon=1e-8):
    """
    Calculates cost and its gradient using Cross Entropy.
    
    (-1 / m) * np.sum(Y * np.log(Y_hat))
    
    Args:
        Y:
            Data labels.
        
        Y_hat:
            Predictions/activations from the output layer.
        
        epsilon: float
            Zero corrector.
    
    Returns:
        cost:
            Cross-Entropy Cost result.
        
        dY_hat:
            Gradient.
    """
    
    m = Y.shape[0]
    
    # make data safe
    Y_hat = np.clip(Y_hat, a_min=epsilon, a_max=(1 - epsilon))
    
    # calc cost
    cost = (-1 / m) * np.nansum(Y * np.log(Y_hat))
    cost = np.squeeze(cost)
    
    # calc gradient
    dY_hat = -Y / Y_hat
    
    return cost, dY_hat
