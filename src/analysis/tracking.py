'''
Tracks regions of interest in the single cell videos.
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
import matplotlib.pyplot as plt

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
        
        '''
        graph_matrices = np.load(distance_file_path)
        for i, matrix in enumerate(graph_matrices):
            eigen_vals, eigen_vecs = spectral_decomposition(matrix)
            eigen_vals, eigen_vecs, sorted_indices = sort_eigens(
                eigen_vals, eigen_vecs)
            # Line 133-134: Remove frame loop
            # Line 135: for j in sorted_indices
        '''
        means, covars = inter['means'], inter['covars']
        color = [np.random.randint(256), np.random.randint(256), np.random.randint(256)]
        #for frame in range(len(means)):
        for frame in range(1):
            for j, mean in enumerate(means[frame]):
                x_diff = std_count * math.sqrt(covars[frame][j][0][0])
                y_diff = std_count * math.sqrt(covars[frame][j][1][1])
                x_bounds = [int(mean[0] - x_diff), int(mean[0] + x_diff)] #means[i][j][0]
                y_bounds = [int(mean[1] - y_diff), int(mean[1] + y_diff)] #means[i][j][1]
            
            vid[frame][x_bounds[0]:x_bounds[1], y_bounds[0], :] = color
            vid[frame][x_bounds[0]:x_bounds[1], y_bounds[1], :] = color
            vid[frame][x_bounds[0], y_bounds[0]:y_bounds[1], :] = color
            vid[frame][x_bounds[1], y_bounds[0]:y_bounds[1], :] = color
            plt.imshow(vid[frame])
            plt.show()



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
        description='Tracks regions of interest in the single cell videos.')
    parser.add_argument('-d', '--distances', required=True, 
                        help='Weighted graph adjacency matrix or a directory'
                             + '  of matrices.')
    parser.add_argument('-i', '--intermediates', required=True,
                        help='GMM intermediate file (.npz) or a directory of' 
                              + ' files.')

    parser.add_argument('-v', '--videos', required=True,
                        help='Single cell video (.avi) or a directory of' 
                              + ' videos.')
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
    regions_of_interest(args['distances'], args['intermediates'], 
                        args['videos'], args['output'])
    
if __name__ == '__main__':
    main()
