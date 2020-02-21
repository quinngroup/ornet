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

import cv2
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .util import spectral_decomposition, sort_eigens

def regions_of_interest(distances, intermediates, videos, outdir, std_count=3):
    '''
    Draws bounding boxes around frame regions demonstrating
    the highest amounts of variance.

    Parameters
    ----------
    distances: list
        Paths to weighted adjacency matrices (.npy).
    intermediates: list
        Paths to gmm intermediates (.npz).
    videos: list
        Paths to single cell videos (.avi).
    outdir: string
        Path to directory to save output video.
    std_count: int
        Number of standard deviations away from the means
        to draw the bounding box borders.
    '''
    for distance_file_path in tqdm(distances):
        filename = os.path.split(distance_file_path)[-1].split('.')[0]

        for inter_file_path in intermediates:
            if filename in inter_file_path:
                inter = np.load(inter_file_path)

        for vid_file_path in videos:
            video_name  = re.sub('_gray', '', filename)
            if video_name in vid_file_path:
                reader = imageio.get_reader(vid_file_path)
                fps = reader.get_meta_data()['fps']
                size = reader.get_meta_data()['size']
                vid = list(reader)
        
        graph_matrices = np.load(distance_file_path)
        for i, matrix in enumerate(graph_matrices):
            eigen_vals, eigen_vecs = spectral_decomposition(matrix)
            eigen_vals, eigen_vecs, sorted_indices = sort_eigens(
                eigen_vals, eigen_vecs)
            means, covars = inter['means'], inter['covars']
            color = [np.random.randint(256), np.random.randint(256), np.random.randint(256)]
            for j in sorted_indices[:1]:
                x_diff = std_count * math.sqrt(covars[i][j][0][0])
                y_diff = std_count * math.sqrt(covars[i][j][1][1])
                x_bounds = [int(means[i][j][0] - x_diff), int(means[i][j][0] + x_diff)]
                y_bounds = [int(means[i][j][1] - y_diff), int(means[i][j][1] + y_diff)]

                if x_bounds[0] < 0:
                    x_bounds[0] = 0

                if x_bounds[0] >= size[0]:
                    x_bounds[0] = size[0] - 1;

                if y_bounds[0] < 0:
                    y_bounds[0] = 0

                if y_bounds[1] >= size[1]:
                    y_bounds[1] = size[1] - 1;
                
                vid[i][x_bounds[0]:x_bounds[1], y_bounds[0], :] = color
                vid[i][x_bounds[0]:x_bounds[1], y_bounds[1], :] = color
                vid[i][x_bounds[0], y_bounds[0]:y_bounds[1], :] = color
                vid[i][x_bounds[1], y_bounds[0]:y_bounds[1], :] = color

        writer = cv2.VideoWriter(
            os.path.join(outdir, video_name + '.avi'),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            fps, size
        )

        for frame in vid:
            writer.write(frame)

        reader.close()
        writer.release()

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

    if os.path.isfile(args['distances']):
        args['distances'] = [args['distances']]
    else:
        args['distances'] = [os.path.join(args['distances'], x) 
                         for x in os.listdir(args['distances'])]

    if os.path.isfile(args['intermediates']):
        args['intermediates'] = [args['intermediates']]
    else:
        args['intermediates'] = [os.path.join(args['intermediates'], x) 
                         for x in os.listdir(args['intermediates'])]

    if os.path.isfile(args['videos']):
        args['videos'] = [args['videos']]
    else:
        args['videos'] = [os.path.join(args['videos'], x) 
                         for x in os.listdir(args['videos'])]

    if not os.path.isdir(args['output']):
        sys.exit('Output is not a directory.')

    return args

def main():
    args = parse_cli(sys.argv[1:])
    regions_of_interest(args['distances'], args['intermediates'], 
                        args['videos'], args['output'])
    
if __name__ == '__main__':
    main()
