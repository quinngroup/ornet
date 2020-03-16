'''
A utility module to assist in OrNet analysis operations.
'''

import os

import numpy as np
from tqdm import tqdm
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

def sort_eigens(eigen_vals, eigen_vecs):
    '''
    Sorts the eigenvalues vector and rearranges the columns
    of the eigeavalue matrix to correspond with the sorted
    eigenvalues.

    Parameters
    ----------
    eigen_vals: vector
        Eigenvalues vector to be sorted.
    eigen_vecs: matrix 
        Eigenvectors to be rearranged.

    Returns
    ---------
    sorted_eigen_vals: vector
        Sorted eigen_vals.
    sorted_eigen_vecs: matrix
        Sorted eigen_vecs.
    sorted_indices: list
        The sorted order of indices of the original
        eigen_vals vector.
    '''
    sorted_indices = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    return eigen_vals, eigen_vecs, sorted_indices

def generate_eigens(input_dir, output_dir):
    '''
    Computes the eigenvalues and eigenvectors of
    distance matrices.

    Parameters
    ---------
    input_dir: string
        Path to the distance matrices. (.npy)
    ouput_dir: string
        Path to save the resulting eigen information.

    Returns
    -------
    NoneType object
    '''

    for filename in os.listdir(input_dir):
        print(filename)
        vid_title = filename.split('.')[0]
        vid = np.load(os.path.join(input_dir, filename))
        eigen_vals = []
        eigen_vecs = []
        for frame in tqdm(vid):
            eigen_val, eigen_vec = spectral_decomposition(frame)
            eigen_vals.append(eigen_val)
            eigen_vecs.append(eigen_vec)

        np.savez(os.path.join(output_dir, vid_title + '.npz'),
                 eigen_vals=eigen_vals, eigen_vecs=eigen_vecs)
