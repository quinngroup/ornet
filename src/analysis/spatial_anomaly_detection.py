'''
Draws bounding boxes around components demonstrating simalar
anomalous behavior.
'''

import os

import imageio
import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(eigen_vecs, k=3):
    '''
    K-means clustering of the Guassian mixture
    components based on their eigenvector values.

    Parameters
    ----------
    eigen_vecs: NumPy array (NxMxM)
    '''
    
    labels = []
    for i in range(eigen_vecs.shape[0]):
        kmeans = KMeans(n_clusters=k).fit(eigen_vecs[i])
        labels.append(kmeans.labels_)

    return labels

def draw_bounding_boxes(frames, means, covars, labels):
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
    labels: list
        Cluster membership labels of the components.
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
    num_of_clusters = len(np.unique(labels))

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
    num_of_clusters = 3

    eigen_data = np.load(eigen_data_path)
    eigen_vals, eigen_vecs = eigen_data['eigen_vals'], eigen_data['eigen_vecs']
    inter = np.load(inter_path)
    means, covars = inter['means'], inter['covars']
    with imageio.get_reader(vid_path) as reader:
        frames = list(reader)

    labels = spectral_clustering(eigen_vecs, k=num_of_clusters)
    #draw_bounding_boxes(frames, means, covars, labels)

if __name__ == '__main__':
    main()
