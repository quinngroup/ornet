'''
Generates a supervsied learning dataset of the
eigenspectrum data using fourier transforms.
'''

import os
import sys
import csv
import argparse

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft

from ornet.analysis.util import sort_eigens

def generate_frequency_dataset(eigen_dir_path, outfile_path, 
                               label, plot_dir=None, k=None):
    '''
    Applies to Fourier Transforms to the leading
    eigenvalues, and uses the frequencies as features
    to construct a supervised learning dataset.

    Parameters
    ----------
    eigen_dir_path: string
        Path to the directory containing eigen data (.npz).
    outfile_path: string
        Path to save the dataset (.csv).
    label: string
        Class label for the instances.
    plot_dir: string
        If a path is given the frequency plots
        corresponding to each eigenvalue time series
        is saved at the given directory path.
    k: int
        First k leading eigenvalues. Default is
        all eigenvalues.
    
    Returns
    -------
    NoneType object
    '''

    with open(outfile_path, 'a+') as fp:
        writer = csv.writer(fp)
        headers = ['frequency_' + str(x) for x in np.arange(100)] 
        headers.append('class')
        writer.writerow(headers)

        for file_path in tqdm(os.listdir(eigen_dir_path)):
            file_name = file_path.split('.')[0]
            data = np.load(os.path.join(eigen_dir_path, file_path))
            eigen_vals, eigen_vecs = data['eigen_vals'], data['eigen_vecs']

            if(eigen_vals.shape[0] == 200):
                sorted_eigen_vals = np.empty(eigen_vals.shape)
                for i in range(len(eigen_vals)):
                    current_eigen_vals, current_eigen_vecs, sorted_indices = sort_eigens(
                        eigen_vals[i], 
                        eigen_vecs[i]
                    )
                    sorted_eigen_vals[i] = current_eigen_vals

                if k == None or k > sorted_eigen_vals.shape[1]:
                    k = sorted_eigen_vals.shape[1]

                sampling_rate = len(sorted_eigen_vals)
                first_half = len(sorted_eigen_vals) // 2
                magnitude_sum = np.zeros(first_half)

                for i in range(k):
                    #Compute the frequencies and store the first half
                    adjusted_eigen_vals =  sorted_eigen_vals[:, i] \
                                            - np.mean(sorted_eigen_vals[:, i])
                    eigen_fft = fft(adjusted_eigen_vals)
                    freqs = fftfreq(len(sorted_eigen_vals)) * sampling_rate
                    freqs = freqs[:first_half] 

                    #Compute the magnitude of the singal and normalize
                    magnitude = np.abs(eigen_fft) * (1/len(eigen_fft)) 
                    magnitude = magnitude[:first_half]
                    magnitude_sum += magnitude

                    #(Optional) Save the eigenvalue frequency plots
                    if plot_dir != None:
                        sns.set()
                        plot_title = file_name + '_leading_eigenval_' + str(i)
                        plt.title(plot_title)
                        fig = plt.Figure()
                        ax = fig.add_subplot(111)
                        ax.bar(freqs, magnitude, width=1.5)
                        ax.set_xlabel('Frequency')
                        ax.set_ylabel('Magnitude')
                        fig.savefig(os.path.join(plot_dir, plot_title + '.png'))
                        plt.close()

                #Write to CSV
                magnitude_list = list(magnitude_sum)
                magnitude_list.append(label)
                writer.writerow(magnitude_list)

def parse_cli(args):
    '''
    Parses command line arguments.
    
    Parameters
    ----------
    args: list
        Arguments from the command line.

    Returns
    -------
    parsed_args: dict
        Key-value pairs of the argument name the argument.
    '''

    parser = argparse.ArgumentParser(
        description='Generates a supervsied learning dataset of the'
                    + ' eigenspectrum data using fourier transforms.'
    )
    parser.add_argument('-i', '--input_path', required=True,
                        help='Path to the directory containing eigen data' 
                             ' (.npz).')
    parser.add_argument('-l', '--label', required=True,
                        help='Class label for the instances.')
    parser.add_argument('-o', '--outfile', default=os.getcwd(),
                        help='Path to save the dataset (.csv)')
    parser.add_argument('-p', '--plot_dir', default=None,
                        help='Path to a directory to save the frequency'
                             + ' plots.')
    parser.add_argument('-k', '--leading', default=None, 
                        help='Use the K-leading eigenvalues.')
    return vars(parser.parse_args(args))

def main():
    args = parse_cli(sys.argv[1:])
    if args['plot_dir'] and args['leading'] != None:
        generate_frequency_dataset(
            args['input_path'], 
            args['outfile'], 
            args['label'],
            args['plot_dir'],
            int(args['leading'])
        )
    elif args['plot_dir'] != None:
        generate_frequency_dataset(
            args['input_path'], 
            args['outfile'], 
            args['label'],
            plot_dir=args['plot_dir'],
        )
    elif args['leading'] != None:
        generate_frequency_dataset(
            args['input_path'], 
            args['outfile'], 
            args['label'],
            k=int(args['leading'])
        )
    else:
        generate_frequency_dataset(
            args['input_path'], 
            args['outfile'], 
            args['label']
        )

if __name__ == '__main__':
    main()
