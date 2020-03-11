'''
A utility module to assist in OrNet analysis operations.
'''

import numpy as np
from scipy.sparse import csgraph

def compute_similarity(matrix, beta=5):
    '''
    Applies a heat kernel to a divergence matrix to convert
    it to a similarity/affinity matrix.

    Parameters
    ----------
    matrix: numpy array of floats
        divergence matrix to be computed into a similarity matrix.

    Returns
    ----------
    similarity: numpy array of floats
        similarity matrix of the input.
    '''
    return np.exp((-beta * matrix) / np.std(matrix))

def sort_eigens(eigen_vals, eigen_vecs):
    '''
    '''
    sorted_indices = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    return eigen_vals, eigen_vecs, sorted_indices

def spectral_decomposition(matrix):
    '''
    Perform eigendecomposition on a weighted graph matrix,
    by converting 

    Parameters
    ----------
    matrix: 2-d numpy array
        Weighted graph adjacency matrix.

    Returns
    ---------
    eigen_vals: numpy array
        Vector of the graph laplacian eigenvalues.
        Sorted from largest to smallest.
    eigen_vecs: 2-d numpy array
        Matrix of the graph laplacian eigenvectors.
        Sorted from largest to smallest.
    '''

    affinity = compute_similarity(matrix)
    laplacian = csgraph.laplacian(affinity, normed=True)

    return np.linalg.eig(laplacian)
