'''
A module to perform spectral decomposition for downstream
OrnNet analysis tasks.
'''

import argparse
import sys
import os

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph
from tqdm import tqdm

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

    #affinity = compute_similarity(matrix)
    affinity = matrix
    laplacian = csgraph.laplacian(affinity, normed=True)    
    eigen_vals, eigen_vecs = eigh(laplacian)
    eigen_vals, eigen_vecs = np.flip(eigen_vals), np.flip(eigen_vecs, axis=1)
    return eigen_vals, eigen_vecs

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

    for filename in tqdm(os.listdir(input_dir)):
        vid_title = filename.split('.')[0] + '_eigendata'
        vid = np.load(os.path.join(input_dir, filename))
        eigen_vals = []
        eigen_vecs = []
        for frame in vid:
            eigen_val, eigen_vec = spectral_decomposition(frame)
            eigen_vals.append(eigen_val)
            eigen_vecs.append(eigen_vec)

        np.savez_compressed(os.path.join(output_dir, vid_title + '.npz'),
                 eigen_vals=eigen_vals, eigen_vecs=eigen_vecs)

def parse_cli(args):
    '''
    Parses command line arguments.

    Parameters
    ----------
    args: list
        Unparsed keys and values.

    Returns
    -------
    parsed_args: dict
        Parsed key-value pairs.
    '''

    parser = argparse.ArgumentParser(
        description='Reads in distance/affinity matrices to perform spectral'
                    + ' decomposition.'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the distance matrices. (.npy)')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to save the resulting eigen information.')

    return vars(parser.parse_args(args))

def main():
    args = parse_cli(sys.argv[1:])
    input_dir_path = args['input']
    output_dir_path = args['output']

    generate_eigens(input_dir_path, output_dir_path)


if __name__ == '__main__':
    main()
