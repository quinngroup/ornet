'''
Generates supervised learning datasets of different
cell types.
'''

import os
import sys
import csv
import argparse

import numpy as np
from tqdm import tqdm
from scipy.sparse import csgraph

from .util import spectral_decomposition, sort_eigens

def manual_dataset(videos, label, outdir):
    '''
    Accepts a directory of cell distance files and creates a 
    supervised learning dataset to classify each cell type 
    by eigenspectrum features. 
    '''

    output = os.path.join(outdir, 'dataset.csv')
    with open(output, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(('mean_diff', 'mean_std', 'mean_abs_change', 
                         'mean_condition_number', 'class'))
        for vid_path in tqdm(videos):
            vid = np.load(vid_path)
            eigenspectrum = [spectral_decomposition(x) for x in vid]
            for frame in eigenspectrum:
                eigen_vals, eigen_vecs = frame[0], frame[1]

            diffs = []
            stds = []
            conds = []
            sorted_vals = sorted(eigen_vals, reverse=True)
            sorted_vals = sorted_vals[:10]
            smallest = eigen_vals[0]
            for val in eigen_vals:
                if (0 < val < smallest) or (smallest == 0 and val != 0):
                    smallest = val
            #diffs.append(np.max(frame) - smallest)
            diffs.append(np.max(sorted_vals) - np.min(sorted_vals))
            stds.append(np.std(sorted_vals))
            conds.append(np.max(eigen_vals) / smallest)
            
            mean_abs_change = 0
            for i in range(len(eigen_vals) - 1):
                mean_abs_change += (
                    abs(np.max(eigen_vals[i]) - np.max(eigen_vals[i + 1])) 
                    / (len(eigen_vals) - 1)
                )
            
            writer.writerow((np.mean(diffs), np.mean(stds), mean_abs_change,
                             np.mean(conds), label))

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
        description='Generates supervised learning datasets of different cell'
                    + ' types.')
    #Required
    parser.add_argument('-i', '--input', required=True, 
                        help='Weighted graph adjacency matrix or a directory'
                             + '  of matrices.')
    parser.add_argument('-l', '--label', required=True,
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

def main():
    args = parse_cli(sys.argv[1:])
    manual_dataset(args['input'], args['label'], args['output'])
    
if __name__ == '__main__':
    main()
