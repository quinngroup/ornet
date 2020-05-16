'''
Draws bounding boxes around components demonstrating simalar
anomalous behavior.
'''

import os
import math

import imageio
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

def spectral_clustering(eigen_vecs, k=3):
    '''
    K-means clustering of the Guassian mixture
    components based on their eigenvector values.

    Parameters
    ----------
    eigen_vecs: NumPy array (NxMxM)
        Eigenvector matrix. N represents the number of
        frames in the corresponding video, M is the
        number of mixture components.
    k: int
        Number of clusters for the spectral clustering
        algorithm.
    '''
    
    labels = []
    for i in range(eigen_vecs.shape[0]):
        kmeans = KMeans(n_clusters=k).fit(eigen_vecs[i])
        labels.append(kmeans.labels_)

    return labels

def euclidean_distance(x, y):
    '''
    Computes the euclidean distance between two vectors.

    Parameters
    ----------
    x,y: NumPy array (m,)
        Eigenvector of length m, where m is the number
        of Gaussian mixture components.

    Returns
    -------
    distance: float
        Euclidean distance between two vectors.
    '''

    return math.sqrt(np.sum((x - y)**2))

def absolute_distance_traveled(eigen_vecs):
    '''
    Computes the absolute distance traveled
    of each mixture component.

    Parameters
    ----------
    eigen_vecs: NumPy array (NxMxM)
        Eigenvector matrix. N represents the number of
        frames in the corresponding video, M is the
        number of mixture components.

    Returns
    -------
    distances: NumPy array(m,)
        Vector of distances of length m, where denotes
        the number of mixture components.
    '''

    distances = np.zeros(eigen_vecs.shape[1], dtype=np.float)
    for i in range(eigen_vecs.shape[0] - 1):
        for j in range(eigen_vecs.shape[1]):
            #distances[j] += euclidean_distance(eigen_vecs[i,j], 
            #                                  eigen_vecs[i + 1,j])
            distances[j] += euclidean(eigen_vecs[i,j], eigen_vecs[i + 1,j])

    return distances


def draw_bounding_boxes(frames, means, covars, eigen_vecs, k):
    '''
    Draws bounding boxes around the mixture component
    regions demonstrating the most variance. (?)

    Parameters
    ----------
    frames: list
        Video frames to be drawn on.
    means: NumPy array (NxMx2)
        Pixel coordinates corresponding to the mixture
        component means. N is the number of video frames,
        M the number of mixture components, and 2 denotes
        the 2D pixel coordinate.
    covars: NumPy array (NxMx2x2)
        Covariance matrices of the guassian mixture 
        components. N is the number of video frames,
        M is the number of mixture components, and 2x2
        denotes the covariance matrix.
    eigen_vecs: NumPy array (NxMxM)
        Eigenvector matrix. N represents the number of
        frames in the corresponding video, M is the
        number of mixture components.
    k: int
        Number of clusters for the spectral clustering
        algorithm.
    '''
    
    '''
        - Randomly generate box colors.
        - Draw bounding boxes for every component.
            - Or most significant variance could be determined
              by computing which components "travel" the most
              in euclidean space.
                - Largest distances traveled
                - Z-Score based, with a threshold of 2.
    '''
    #labels = spectral_clustering(eigen_vecs, k=k)
    distances = absolute_distance_traveled(eigen_vecs)   
    descending_distances_indices = np.flip(np.argsort(distances))
    print(np.flip(np.argsort(distances)))

def main():
    '''
        - Load in the eigendata.
        - Load in intermediates data.
        - Load in a video.

        - Cluster the eigendata.
        - Use the intermediates to draw bounding boxes.
            - Color cluster members the same box.
    '''

    
    eigen_data_path = \
    '/extrastorage/ornet/Eigenspectrum/Eigendata/DsRed2-HeLa_3_15_LLO1part1_1.npz'   
    inter_path = \
        '/extrastorage/ornet/distances/Hellinger Distances/intermediates/DsRed2-HeLa_3_15_LLO1part1_1.npz'
    vid_path = \
        '/home/marcus/Desktop/Anomaly-Detection/outputs/singles/DsRed2-HeLa_3_15_LLO1part1_1.avi'
    k = 3

    eigen_data = np.load(eigen_data_path)
    eigen_vals, eigen_vecs = eigen_data['eigen_vals'], eigen_data['eigen_vecs']
    inter = np.load(inter_path)
    means, covars = inter['means'], inter['covars']
    with imageio.get_reader(vid_path) as reader:
        frames = list(reader)

    draw_bounding_boxes(frames, means, covars, eigen_vecs, k)

if __name__ == '__main__':
    main()
