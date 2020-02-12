'''
This script creates vizualizations of the eigenspectrum 
data for analysis.
'''
#Author: Marcus Hill

import os
import sys
import csv
import argparse

from tqdm import tqdm
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
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

def eigen_decomposition(matrix):
    '''
    Perform eigendecomposition on the input graph matrix.

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
    laplacian = csgraph.laplacian(affinity)
    eigen_vals, eigen_vecs = sla.eigh(laplacian)

    sorted_indices = list(range(len(eigen_vals)).__reversed__())
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]  

    return eigen_vals, eigen_vecs

def eigenspectrum_plot(args):
    '''
    Plots the eigenspectrum data across all video frames.

    Parameters
    ----------
    args: dict
        Parsed cli arguments. Details the input video(s)
        path(s) and output directory.

    Returns
    ----------
    NoneType object
    '''

    progress_bar = tqdm(total=len(args['input']))
    for vid in args['input']:
        frames = np.load(vid)
        vid_eigenvals = []
        for graph_matrix in frames:
            eigen_vals, eigen_vecs = eigen_decomposition(graph_matrix)
            vid_eigenvals.append(eigen_vals)

        vid_name = os.path.split(vid)[-1].split('.')[0]
        vid_eigenvals = np.array(vid_eigenvals)
        plt.plot(vid_eigenvals)
        plt.suptitle(vid_name)
        plt.savefig(os.path.join(args['output'], vid_name))
        progress_bar.update()

    progress_bar.close()

def generate_dataset(args):
    '''
    Accepts a directory of cell distance files and creates a 
    supervised learning dataset to classify each cell type 
    by eigenspectrum features. 
    '''

    output = os.path.join(args['output'], 'dataset.csv')
    with open(output, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(('mean_diff', 'mean_std', 'mean_abs_change', 
                         'mean_condition_number', 'class'))
        for vid_name in os.listdir(args['input']):
            vid = np.load(os.path.join(args['input'], vid_name))
            affinities = [compute_similarity(x) for x in vid]
            laplacians = [csgraph.laplacian(x) for x in affinities]
            eigens = [sla.eigh(x) for x in laplacians]
            eigvals = []
            eigvecs = []
            for eigen in eigens:
                eigvals.append(eigen[0])
                eigvecs.append(eigen[1])

            eigvals = np.array(eigvals)
            eigvecs = np.array(eigvecs)

            diffs = []
            stds = []
            conds = []
            for frame in eigvals:
                sorted_vals = sorted(frame, reverse=True)
                sorted_vals = sorted_vals[:10]
                smallest = frame[0]
                for val in frame:
                    if (0 < val < smallest) or (smallest == 0 and val != 0):
                        smallest = val
                #diffs.append(np.max(frame) - smallest)
                diffs.append(np.max(sorted_vals) - np.min(sorted_vals))
                stds.append(np.std(sorted_vals))
                conds.append(np.max(frame) / smallest)
            
            mean_abs_change = 0
            for i in range(len(eigvals) - 1):
                mean_abs_change += (
                    abs(np.max(eigvals[i]) - np.max(eigvals[i + 1])) 
                    / (len(eigvals) - 1)
                )
            
            writer.writerow((np.mean(diffs), np.mean(stds), mean_abs_change,
                             np.mean(conds), args['class']))

            '''
            maxs = [np.max(x) for x in eigvals]
            plt.plot(maxs)
            plt.show()
            '''

def parse_cli(cli_args):
    '''
    Parses the arguments passed in from the command line.

    Parameters
    ----------
    cli_args: list
        Arguments stored in sys.argv.

    Returns:
    ----------
    args: dict
        Key value option-argument pairs.
    '''
    parser = argparse.ArgumentParser(
        description='This script takes a numpy array, or a directory of arrays'
                    + ' , of cell distance files and can generate either a'
                    + ' dataset, or output the eigenspectrum.')
    #Required
    parser.add_argument('-i', '--input', required=True, 
                        help='Weighted graph adjacency matrix or a directory'
                             + '  of matrices.')

    #optional
    parser.add_argument('-c', '--class',
                        help='Class label for the cells'
                             + ' (e.g. control, llo, mdivi).')
    parser.add_argument('-o', '--output', 
                        default=os.getcwd(), 
                        help='Output directory.')
    args = vars(parser.parse_args(cli_args))

    if os.path.isfile(args['input']):
        args['input'] = [args['input']]
    else:
        args['input'] = [os.path.join(args['input'], x) 
                         for x in os.listdir(args['input'])]

    if not os.path.isdir(args['output']):
        sys.exit('Output is not a directory.')

    return args

if __name__ == '__main__':
    args = parse_cli(sys.argv[1:])
    eigenspectrum_plot(args)
