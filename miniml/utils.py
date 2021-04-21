#  Created byMartin.cz

import math
import numpy as np


def shuffle_data(X, Y, seed=None):
    """
    Shuffles data by random permutations.
    
    Args:
        X: np.ndarray
            Input dataset of shape (m, ...).
        
        Y: np.ndarray
            Output dataset (m, ...).
        
        seed: int or None
            Random seed for permutations.
    
    Returns:
            Shuffled datasets (X, Y).
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
        X: np.ndarray
            Input dataset of shape (m, ...).
        
        Y: np.ndarray
            Output dataset of shape (m, ...).
        
        size: int
            Size of a mini-batch. Should be 64, 128, 256 or 512. If set to
            zero, no mini-batches will be created.
        
        seed: int or None
            Random seed for mini-batches creation.
    
    Returns:
        List of mini-batches as [(X_batch, Y_batch),]
    """
    
    # check if needed
    if not size:
        return [(X, Y)]
    
    # set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # get batch count
    m = X.shape[0]
    count = int(math.floor(m / size))
    batches = []
    
    # shuffle data
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    
    # create full-size batches
    for k in range(0, count):
        batch_X = shuffled_X[k * size: (k + 1) * size]
        batch_Y = shuffled_Y[k * size: (k + 1) * size]
        batches.append((batch_X, batch_Y))
    
    # create remaining batch
    if m % size != 0:
        batch_X = shuffled_X[count * size: m]
        batch_Y = shuffled_Y[count * size: m]
        batches.append((batch_X, batch_Y))
    
    return batches


def im2col_idxs(shape, kernel, pad, stride):
    """
    Adapted from Stamford CS231N course: https://cs231n.github.io/
    
    Args:
        shape: (int, int, int)
            Input shape as (m, c_in, h_in, w_in).
        
        kernel: (int, int)
            Filter kernel size as (f_h, f_w)
        
        pad: (int, int, int, int)
            Padding as (p_t, p_b, p_l, p_r).
        
        stride: (int, int)
            Stride as (s_h, s_w).
    """
    
    m, c_in, h_in, w_in = shape
    f_h, f_w = kernel
    p_t, p_b, p_l, p_r = pad
    s_h, s_w = stride
    
    h_out = (h_in + p_t + p_b - f_h) // s_h + 1
    w_out = (w_in + p_l + p_r - f_w) // s_w + 1

    i0 = np.repeat(np.arange(f_h), f_w)
    i0 = np.tile(i0, c_in)
    i1 = s_h * np.repeat(np.arange(h_out), w_out)
    
    j0 = np.tile(np.arange(f_w), f_h * c_in)
    j1 = s_w * np.tile(np.arange(w_out), h_out)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c_in), f_h * f_w).reshape(-1, 1)
    
    return k, i, j


def im2col(X, kernel, pad, stride):
    """
    Adapted from Stamford CS231N course: https://cs231n.github.io/
    
    Args:
        X: np.array
            Input of shape (m, c_in, h_in, w_in).
        
        kernel: (int, int)
            Filter kernel size as (f_h, f_w)
        
        pad: (int, int, int, int)
            Padding as (p_t, p_b, p_l, p_r).
        
        stride: (int, int)
            Stride as (s_h, s_w).
    
    Returns:
        (f_h * f_w * c_in, h_out, w_out * m)
    """
    
    m, c_in, h_in, w_in = X.shape
    f_h, f_w = kernel
    p_t, p_b, p_l, p_r = pad
    
    padded = np.pad(
        array = X,
        pad_width = ((0, 0), (0, 0), (p_t, p_b), (p_l, p_r)),
        mode = 'constant')
    
    k, i, j = im2col_idxs(X.shape, kernel, pad, stride)
    cols = padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(f_h * f_w * c_in, -1)
    
    return cols


def col2im(cols, shape, kernel, pad, stride):
    """
    Adapted from Stamford CS231N course: https://cs231n.github.io/
    
    Args:
        cols:
            (f_h * f_w * c_in, h_out, w_out * m)
        
        shape: (int, int, int)
            Input shape as (m, c_in, h_in, w_in).
        
        kernel: (int, int)
            Filter kernel size as (f_h, f_w)
        
        pad: (int, int, int, int)
            Padding as (p_t, p_b, p_l, p_r).
        
        stride: (int, int)
            Stride as (s_h, s_w).
    
    Returns:
        (m, c_in, h_in, w_in)
    """
    
    m, c_in, h_in, w_in = shape
    f_h, f_w = kernel
    p_t, p_b, p_l, p_r = pad
    p_h = h_in + p_t + p_b
    p_w = w_in + p_l + p_r
    
    padded = np.zeros((m, c_in, p_h, p_w), dtype=cols.dtype)
    k, i, j = im2col_idxs(shape, kernel, pad, stride)
    
    cols_reshaped = cols.reshape(c_in * f_h * f_w, -1, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(padded, (slice(None), k, i, j), cols_reshaped)
    output = padded[:, :, p_t:p_t+h_in, p_l:p_l+w_in]
    
    return output
