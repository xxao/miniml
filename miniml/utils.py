#  Created byMartin.cz

import math
import numpy as np


def shuffle_data(X, Y, seed=None):
    """
    Shuffles data by random permutations.
    
    Args:
        X:
            Input dataset.
        
        Y:
            Output dataset.
        
        seed: int or None
            Random seed for permutations.
    
    Returns:
            XShuffled data sets X and Y.
    """
    
    # set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # randomize data order
    idxs = np.random.permutation(len(X))
    X = X[idxs]
    Y = Y[idxs]
    
    return X, Y


def to_categorical(Y, sparse=True):
    """
    Converts arbitrary class vector into indexes or sparse one-hot matrix.
    
    Args:
        Y:
            Class vector.
        
        sparse: bool
            If set to True, sparse on-hot matrix is returned instead of just
            indexes.
        
        Returns:
            Y:
                Converted vector.
            
            mappings: {any: int}
                Mappings from category to unique index.
    """
    
    # get sorted unique categories
    mappings = {cat: i for i, cat in enumerate(sorted(set(Y)))}
    
    # convert data to category indexes
    idxs = np.array([mappings[y] for y in Y])
    if not sparse:
        return idxs, mappings
    
    # make sparse matrix
    mat = np.eye(len(mappings))[idxs]
    
    return mat, mappings


def make_mini_batches(X, Y, size=64, seed=None):
    """
    Initializes random mini-batches from X, Y.
    
    Args:
        X:
            Input dataset.
        
        Y:
            Output dataset.
        
        size: int
            Size of a mini-batch. Should be 64, 128, 256 or 512. If set to
            zero, no mini-batches will be created.
        
        seed: int or None
            Random seed for mini-batches creation.
    
    Returns:
        List of mini-batches as [(X, Y),]
    """
    
    # check if needed
    if not size:
        return [(X, Y)]
    
    # set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # get batch count
    m = X.shape[1]
    count = int(math.floor(m / size))
    batches = []
    
    # shuffle data
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    # create full-size batches
    for k in range(0, count):
        batch_X = shuffled_X[:, k * size: (k + 1) * size]
        batch_Y = shuffled_Y[:, k * size: (k + 1) * size]
        batches.append((batch_X, batch_Y))
    
    # create remaining batch
    if m % size != 0:
        batch_X = shuffled_X[:, count * size: m]
        batch_Y = shuffled_Y[:, count * size: m]
        batches.append((batch_X, batch_Y))
    
    return batches
