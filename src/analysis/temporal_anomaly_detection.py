'''
Anomaly detection of of eigenvalue time-series data.

Approach 1:
Average all of the K eigenvalue time-series at each
time-point. Find the average of all of the averages then
perform z-score thresholding.

Approach 2:
Calculate z-scores for each eigenvalue time-series 
independent of each other, perfrom a vote to declare
a time point anomalous.

Approach 3:
Adaptive thresholding that uses a sliding window to
calculate a moving average which z-scores are computed
with respect to.
'''

import os
import sys
import argparse

import numpy as np
import seaborn as sns
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

    sns.set()
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

def approach_one(eigen_vals, k=10, threshold=2):
    '''
    Average all of the K eigenvalue time-series at each
    time-point. Find the average of all of the averages then
    perform z-score thresholding.

    Parameters
    ----------
    eigen_vals: numpy array (N x K)
       N eigenvalue vectors of length K, where
       N is the number of frames in the corresponding
       video and K is the number of mixture components.

    Returns
    -------
    NoneType object
    '''
    
    eigen_vals_avgs = [np.mean(x) for x in eigen_vals]
    #eigen_vals_avgs = [np.mean(x[:k]) for x in eigen_vals]
    mean_of_avgs = np.mean(eigen_vals_avgs)
    std_of_avgs = np.std(eigen_vals_avgs)
    z_scores = [(x - mean_of_avgs) / std_of_avgs for x in eigen_vals_avgs]
    plot(eigen_vals[:,:k], z_scores, 'Raw Z-Score Plot')

    signals = np.empty(shape=(len(z_scores),), dtype=np.float)
    for i, score in enumerate(z_scores):
        if score > threshold:
            signals[i] = 1
        elif score < threshold * -1:
            signals[i] = -1
        else:
            signals[i] = 0
    plot(eigen_vals[:,:k], signals, 'Raw Signal Plot')

def temporal_anomaly_detection(vid_name, eigen_vals, outdir_path, k=10, 
                   window=20, threshold=2): #10, 2
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
        Path to a directory to save the plots.
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
    '''
    eigen_vals_avgs = [np.mean(x) for x in eigen_vals]
    moving_avgs = np.empty(shape=(eigen_vals.shape[0],), dtype=np.float)
    moving_stds = np.empty(shape=(eigen_vals.shape[0],), dtype=np.float)
    z_scores = np.empty(shape=(eigen_vals.shape[0],), dtype=np.float)
    signals = np.empty(shape=(eigen_vals.shape[0],), dtype=np.float)

    moving_avgs[:window] = 0
    moving_stds[:window] = 0
    z_scores[:window] = 0
    for i in range(window, moving_avgs.shape[0]):
        moving_avgs[i] = np.mean(eigen_vals_avgs[i - window:i])
        moving_stds[i] = np.std(eigen_vals_avgs[i - window:i])
        z_scores[i] = (eigen_vals_avgs[i] - moving_avgs[i]) / moving_stds[i]

    for i, score in enumerate(z_scores):
        if score > threshold:
            signals[i] = 1
            print(i)
        elif score < threshold * -1:
            signals[i] = -1
            print(i)
        else:
            signals[i] = 0

    #title = vid_name + ' Signals Plot'
    title = ''
    plot(eigen_vals[:,:k], signals, title, False, outdir_path) #True to save


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

    return vars(parser.parse_args(input_args))

def main():
    '''
    eigen_data_path = \
        '/extrastorage/ornet/Eigenspectrum/Eigendata/DsRed2-HeLa_3_15_LLO1part1_1.npz'
    eigen_vals, eigen_vecs = data_loader(eigen_data_path)
    '''
    
    args =  parse_cli(sys.argv[1:])
    for vid_path in os.listdir(args['input']):
        vid_name = os.path.split(vid_path)[1].split('.')[0]
        eigen_data_path = os.path.join(args['input'], vid_path)
        eigen_vals, eigen_vecs = data_loader(eigen_data_path)   
        temporal_anomaly_detection(vid_name, eigen_vals, args['outdir'])

    #approach_one(eigen_vals)

if __name__ == '__main__':
    main()
