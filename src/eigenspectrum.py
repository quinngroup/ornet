'''
This script creates vizualizations of the eigenspectrum 
data for analysis.
'''
#Author: Marcus Hill

import os
import re
import sys
import csv
import math
import argparse

import imageio
import numpy as np
from tqdm import tqdm
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

def sort_eigens(eigen_vals, eiegen_vecs):
    '''
    '''
    sorted_indices = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    return eigen_vals, eigen_vecs

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
    #eigen_vals, eigen_vecs = sla.eigh(laplacian)
    #sorted_indices = list(range(len(eigen_vals)).__reversed__())

    eigen_vals, eigen_vecs = np.linalg.eig(laplacian)
    return eigen_vals, eigen_vecs

def eigenspectrum_plot(vids, outdir, max_vals=10):
    '''
    Plots the eigenspectrum data across all video frames,
    and save the results.

    Parameters
    ----------
    vids: list
        Path(s) to the input video(s).
    outdir: string
        Path to the directory where the plots should be
        saved.

    Returns
    ----------
    NoneType object
    '''
    progress_bar = tqdm(total=len(vids))
    for vid in vids:
        frames = np.load(vid)
        vid_name = os.path.split(vid)[-1].split('.')[0]
        vid_eigenvals = []
        for graph_matrix in frames:
            eigen_vals, eigen_vecs = eigen_decomposition(graph_matrix)
            eigen_vals, eigen_vecs = sort_eigens(eigen_vals, eigen_vecs)
            vid_eigenvals.append(eigen_vals[:max_vals])

        plt.suptitle(vid_name)
        plt.plot(vid_eigenvals)
        plt.savefig(os.path.join(outdir, vid_name))
        progress_bar.update()

    progress_bar.close()

def regions_of_interest(distances, intermediates, videos, outdir, std_count=3):
    '''
    '''
    for distance_file_path in distances:
        filename = os.path.split(distance_file_path)[-1].split('.')[0]

        for inter_file_path in intermediates:
            if filename in inter_file_path:
                inter = np.load(inter_file_path)

        for vid_file_path in videos:
            video_name  = re.sub('_gray', '', filename)
            if video_name in vid_file_path:
                vid = list(imageio.get_reader(vid_file_path))
        
        #Write a for loop iterating through
        #graph_matrices = np.load(distance_file_path)
        means, covars = inter['means'], inter['covars']
        color = [np.random.randint(256), np.random.randint(256), np.random.randint(256)]
        #for frame in range(len(means)):
        for frame in range(1):
            for j, mean in enumerate(means[frame]):
                x_diff = std_count * math.sqrt(covars[frame][j][0][0])
                y_diff = std_count * math.sqrt(covars[frame][j][1][1])
                x_bounds = [int(mean[0] - x_diff), int(mean[0] + x_diff)]
                y_bounds = [int(mean[1] - y_diff), int(mean[1] + y_diff)]
            
            vid[frame][x_bounds[0]:x_bounds[1], y_bounds[0], :] = color
            vid[frame][x_bounds[0]:x_bounds[1], y_bounds[1], :] = color
            vid[frame][x_bounds[0], y_bounds[0]:y_bounds[1], :] = color
            vid[frame][x_bounds[1], y_bounds[0]:y_bounds[1], :] = color
            plt.imshow(vid[frame])
            plt.show()

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
        for vid_name in os.listdir(args['distances']):
            vid = np.load(os.path.join(args['distances'], vid_name))
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
    parser.add_argument('-d', '--distances', required=True, 
                        help='Weighted graph adjacency matrix or a directory'
                             + '  of matrices.')

    #optional
    parser.add_argument('-i', '--intermediates',
                        help='GMM intermediate file (.npz) or a directory of' 
                              + ' files.')

    parser.add_argument('-v', '--videos',
                        help='Single cell video (.avi) or a directory of' 
                              + ' videos.')
    parser.add_argument('-c', '--class',
                        help='Class label for the cells'
                             + ' (e.g. control, llo, mdivi).')
    parser.add_argument('-o', '--output', 
                        default=os.getcwd(), 
                        help='Output directory.')
    args = vars(parser.parse_args(cli_args))

    #Check whether distances is a file or directory
    if os.path.isfile(args['distances']):
        args['distances'] = [args['distances']]
    else:
        args['distances'] = [os.path.join(args['distances'], x) 
                         for x in os.listdir(args['distances'])]


    #Check whether intermediates is a file or a directory
    if os.path.isfile(args['intermediates']):
        args['intermediates'] = [args['intermediates']]
    else:
        args['intermediates'] = [os.path.join(args['intermediates'], x) 
                         for x in os.listdir(args['intermediates'])]

    #Check whether videos is a file or a directory
    if os.path.isfile(args['videos']):
        args['videos'] = [args['videos']]
    else:
        args['videos'] = [os.path.join(args['videos'], x) 
                         for x in os.listdir(args['videos'])]

    #Check whether output is a directory or not.
    if not os.path.isdir(args['output']):
        sys.exit('Output is not a directory.')

    return args

def main():
    args = parse_cli(sys.argv[1:])
    #eigenspectrum_plot(args['distances'], args['output'])
    regions_of_interest(args['distances'], args['intermediates'], 
                        args['videos'], args['output'])

if __name__ == '__main__':
    main()
