'''
Plots the eigenspectrum of the graphs that correspond with
every video frame, and saves the "eigendata".
'''

import os
import sys
import argparse

import numpy as np
import seaborn as sns #Add dependencies to requirements.txt
from tqdm import tqdm
import matplotlib.pyplot as plt

from ornet.analysis.util import sort_eigens, \
    spectral_decomposition, generate_eigens

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
    input_dir = os.path.split(vids[0])[0]
    eigendata_dir_path = os.path.join(outdir, 'Eigendata')
    plot_dir_path = os.path.join(outdir, 'Plots')
    os.makedirs(eigendata_dir_path)
    os.makedirs(plot_dir_path)
    for vid in vids:
        frames = np.load(vid)
        vid_name = os.path.split(vid)[-1].split('.')[0]
        plot_eigenvals = []
        for graph_matrix in frames:
            current_vals, current_vecs = spectral_decomposition(graph_matrix)
            plot_eigenvals.append(current_vals[:max_vals])

        #Save Plots
        plt.suptitle(vid_name)
        ax = plt.subplot(111)
        ax.plot(plot_eigenvals)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Magnitude')
        plt.savefig(os.path.join(plot_dir_path, vid_name))
        progress_bar.update()

    #Save Eigendata
    generate_eigens(input_dir, eigendata_dir_path)

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
        description='Generates eigenspectrum plots from cell distance '
                    + 'files (.npy), and saves the plots and "eigendata".'
                    + ' The eigendata is saved in a numpy zip with the'
                    + ' keywords "eigen_vals" and "eigen_vecs".'
    )
    parser.add_argument('-i', '--input', required=True, 
                        help='Distance file or files (.npy).')
    parser.add_argument('-o', '--output', 
                        default=os.getcwd(), 
                        help='Directory to save the plots.')
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
