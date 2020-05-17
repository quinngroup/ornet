'''
Draws bounding boxes around components demonstrating simalar
anomalous behavior.
'''

import os
import math

import imageio
import numpy as np
from tqdm import tqdm
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

    Returns
    -------
    labels: list (NxM)
        Cluster membership labels of the mixture
        components. N represents the number of 
        frames, and M is the number of mixture
        components.
    '''
    
    labels = []
    for i in range(eigen_vecs.shape[0]):
        kmeans = KMeans(n_clusters=k).fit(eigen_vecs[i])
        labels.append(kmeans.labels_)

    return labels

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
            distances[j] += euclidean(eigen_vecs[i,j], eigen_vecs[i + 1,j])

    return distances


def draw_bounding_boxes(frames, means, covars, eigen_vecs, k, fps, size, std_threshold=3):
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
    out_vid_path = '/home/marcus/Desktop/bounding_box_example.mp4'
    labels = spectral_clustering(eigen_vecs, k=k)
    box_colors = {}
    for i in range(k):
        box_colors[i] = np.random.randint(256, size=(3,))
    
    distances = absolute_distance_traveled(eigen_vecs)   
    descending_distances_indices = np.flip(np.argsort(distances))

    with imageio.get_writer(out_vid_path, mode='I', fps=1) as writer:
        for i, frame in enumerate(tqdm(frames)):
            for j in descending_distances_indices[:1]:
                x_diff = std_threshold * math.sqrt(covars[i][j][0][0])
                y_diff = std_threshold * math.sqrt(covars[i][j][1][1])
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
                
                color = box_colors[labels[i][j]]
                frames[i][x_bounds[0]:x_bounds[1], y_bounds[0], :] = color
                frames[i][x_bounds[0]:x_bounds[1], y_bounds[1], :] = color
                frames[i][x_bounds[0], y_bounds[0]:y_bounds[1], :] = color
                frames[i][x_bounds[1], y_bounds[0]:y_bounds[1], :] = color
            
            writer.append_data(frames[i])

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
        fps = reader.get_meta_data()['fps']
        size = reader.get_meta_data()['size']

    draw_bounding_boxes(frames, means, covars, eigen_vecs, k, fps=fps, size=size)

if __name__ == '__main__':
    main()
