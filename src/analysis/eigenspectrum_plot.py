'''
Plots the eigenspectrum of the graphs that correspond with
every video frame.
'''

import os
import sys
import argparse

import numpy as np
import seaborn as sns #Add dependencies to requirements.txt
from tqdm import tqdm
import matplotlib.pyplot as plt

from ornet.analysis.util import sort_eigens, spectral_decomposition

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

    sns.set()
    progress_bar = tqdm(total=len(vids))
    for vid in vids:
        frames = np.load(vid)
        vid_name = os.path.split(vid)[-1].split('.')[0]
        vid_eigenvals = []
        for graph_matrix in frames:
            eigen_vals, eigen_vecs = spectral_decomposition(graph_matrix)
            eigen_vals, eigen_vecs, ind = sort_eigens(eigen_vals, eigen_vecs)
            vid_eigenvals.append(eigen_vals[:max_vals])

        plt.suptitle(vid_name)
        ax = plt.subplot(111)
        ax.plot(vid_eigenvals)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Magnitude')
        plt.savefig(os.path.join(outdir, vid_name))
        progress_bar.update()

    progress_bar.close()

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
    parser.add_argument('-i', '--input', required=True, 
                        help='Weighted graph adjacency matrix or a directory'
                             + '  of matrices.')
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

def main():
    args = parse_cli(sys.argv[1:])
    eigenspectrum_plot(args['input'], args['output'])

if __name__ == '__main__':
    main()
