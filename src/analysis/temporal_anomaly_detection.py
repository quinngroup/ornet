'''
Anomaly detection of of eigenvalue time-series data.
'''

import os
import sys
import argparse

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def data_loader(eigen_data_path):
    '''
    Loads the eigendata stored in a numpy zip and returns
    the eigenvalues and eigenvectors.

    Paramters
    ---------
    eigen_data_path: string
        Path to the eigendata numpy zip (.npz) file.

    Returns
    -------
    eigen_vals: numpy array
        Eigenvalues stored in the eigendata zip.
    eigen_vecs: numpy array
        Eigenvectors stored in the eigendata zip.
    '''

    eigen_data = np.load(eigen_data_path)
    eigen_vals, eigen_vecs = eigen_data['eigen_vals'], eigen_data['eigen_vecs']
    return eigen_vals, eigen_vecs

def plot(eigen_vals, z_scores, title, save_fig, outdir_path=None):
    '''
    Plots eigenvalue time-series data, and a
    corresponding z-score curve.
    '''

    plt.style.use('seaborn')
    fig = plt.figure()
    fig.suptitle(title)

    ax = fig.add_subplot(211)
    ax.plot(eigen_vals)
    ax.set_ylabel('Magnitude')

    ax = fig.add_subplot(212)
    ax.plot(z_scores)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Signal')
    if save_fig:
        file_name = os.path.join(outdir_path, title.split(' ')[0])
        plt.savefig(file_name)
    else:
        plt.show()
    
    plt.close()

def temporal_anomaly_detection(vid_name, eigen_vals, outdir_path, k=10, 
                   window=20, threshold=2, save_plot=False): #10, 2
    '''
    Generates a figure comprised of a time-series plot
    of the eigenvalue vectors, and an outlier detection 
    signals plot.

    Parameters
    ----------
    vid_name: string
        Name of the microscopy video.
    eigen_vals: NumPy array (NXM)
        Matrix comprised of eigenvalue vectors. 
        N represents the number of frames in the
        corresponding video, and M is the number of
        mixture components.
    outdir_path: string
        Path to a directory to save the plots and anomalous
        time points.
    k: int
        The number of leading eigenvalues to display.
    window: int
        The size of the window to be used for anomaly 
        detection.
    threshold: float
        Value used to determine whether a signal value
        is anomalous.  

    Returns
    -------
    NoneType object
    '''
    eigen_vals_avgs = [np.mean(x) for x in eigen_vals]
    moving_avgs = np.empty(shape=(eigen_vals.shape[0],), dtype=float)
    moving_stds = np.empty(shape=(eigen_vals.shape[0],), dtype=float)
    z_scores = np.empty(shape=(eigen_vals.shape[0],), dtype=float)
    signals = np.empty(shape=(eigen_vals.shape[0],), dtype=float)

    moving_avgs[:window] = 0
    moving_stds[:window] = 0
    z_scores[:window] = 0
    for i in range(window, moving_avgs.shape[0]):
        moving_avgs[i] = np.mean(eigen_vals_avgs[i - window:i])
        moving_stds[i] = np.std(eigen_vals_avgs[i - window:i])
        z_scores[i] = (eigen_vals_avgs[i] - moving_avgs[i]) / moving_stds[i]

    plot_title = vid_name + ' Signals Plot'
    timepoint_title = vid_name + '.txt'
    with open(os.path.join(outdir_path, timepoint_title), 'w+') as writer:
        for i, score in enumerate(z_scores):
            if score > threshold:
                signals[i] = 1
                writer.write(str(i) + '\n')
            elif score < threshold * -1:
                signals[i] = -1
                writer.write(str(i) + '\n')
            else:
                signals[i] = 0

    plot(eigen_vals[:,:k], signals, plot_title, save_plot, outdir_path) #True to save


def parse_cli(input_args):
    '''
    Parses the command line arguments.

    Parameters
    ----------
    input_args: list
        Arguments to be parsed.

    Returns
    -------
    parsed_args: dict
        Key value pairs of arguments.
    '''
    
    parser = argparse.ArgumentParser(
        description='Anomaly detection of of eigenvalue time-series data.'
    )
    parser.add_argument('-i', '--input', required=True, 
                        help='Input directory of eigendata file (.npz).')
    parser.add_argument('-o', '--outdir', default=os.getcwd(),
                        help='Output directory for plots.')
    parser.add_argument('-w', '--window_size', default=20, type=int,
                        help='Window size for the sliding window.')
    parser.add_argument('-t', '--threshold', default=2, type=float,
                        help='Threshold value for declaring z-scores anomalous.')
    parser.add_argument('--save', default=False, action='store_true',
                        help='To save plots.')

    return vars(parser.parse_args(input_args))

def main():
    args =  parse_cli(sys.argv[1:])
    input_dir_path = args['input']
    output_dir_path = args['outdir']
    window_size = args['window_size']
    threshold = args['threshold']
    save_plot = args['save']

    for vid_path in tqdm(os.listdir(input_dir_path)):
        vid_name = os.path.split(vid_path)[1].split('.')[0].replace('_eigendata', '')
        eigen_data_path = os.path.join(input_dir_path, vid_path)
        eigen_vals, eigen_vecs = data_loader(eigen_data_path)   
        temporal_anomaly_detection(vid_name, eigen_vals, output_dir_path, 
                                   window=window_size, 
                                   threshold=threshold,
                                   save_plot=save_plot)

if __name__ == '__main__':
    main()
