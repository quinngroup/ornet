'''
Draws bounding boxes around components demonstrating simalar
anomalous behavior.
'''

import os
import sys
import math
import argparse

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
        Vector of distances of length m, where m denotes
        the number of mixture components.
    '''

    distances = np.zeros(eigen_vecs.shape[1], dtype=np.float)
    for i in range(eigen_vecs.shape[0] - 1):
        for j in range(eigen_vecs.shape[1]):
            distances[j] += euclidean(eigen_vecs[i,j], eigen_vecs[i + 1,j])

    return distances

def does_overlap(box_one, box_two):
    ''' 
    Determines whether box one intersects box two, or is contained 
    within box two.

    Parameters
    ----------
    box_one, box_two: list (4x2)
       2d coordinates of the box. The ordering is 
       top left, top right, bottom left, and bottom right.

    Returns
    --------
    overlap_status: bool
        Whether the boxes overlap each other.
    '''

    overlap_status = False
    #Top left corner check
    if box_one[0][0] >= box_two[0][0] and box_one[0][0] <= box_two[2][0] \
        and box_one[0][1] >= box_two[0][1] and box_one[0][1] <= box_two[1][1]:
        overlap_status = True
    #Bottom left corner check
    elif box_one[2][0] >= box_two[0][0] and box_one[2][0] <= box_two[2][0] \
        and box_one[2][1] >= box_two[0][1] and box_one[2][1] <= box_two[1][1]:
        overlap_status = True
    #Top right corner check
    elif box_one[1][0] >= box_two[0][0] and box_one[1][0] <= box_two[2][0] \
        and box_one[1][1] >= box_two[0][1] and box_one[1][1] <= box_two[1][1]:
        overlap_status = True
    #Bottom right corner check
    elif box_one[3][0] >= box_two[0][0] and box_one[3][0] <= box_two[2][0] \
        and box_one[3][1] >= box_two[0][1] and box_one[3][1] <= box_two[1][1]:
        overlap_status = True

    return overlap_status

def find_initial_boxes(means, covars, size, 
                       descending_distances_indices, k, std_threshold=3):
    '''
    Finds the first k non-overlapping bounding
    boxes.

    Parameters
    ----------
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
    size: tuple (2,)
        Width and height of the video.
    descending_distances_indices: NumPy array(m,)
        Vector of eigenvector distances in descending sorted
        order. Length is the number of eigenvector rows, 
        which is the number of mixture components.
    std_threshold: float 
        The number of standard deviations to use to compute
        the spatial region of the bounding box. Default is
        three.
    k: int
        Number of boxes to utilize.

    Returns
    -------
    region_indices = list 
       The indices of the first k non-overlapping regions. 
       List size is at most k.
    '''

    j = 0
    initial_boxes = []
    region_indices = []
    while j < descending_distances_indices.shape[0] \
          and len(initial_boxes) < k:
        region = descending_distances_indices[j]
        x_diff = std_threshold * math.sqrt(covars[0][region][0][0])
        y_diff = std_threshold * math.sqrt(covars[0][region][1][1])
        x_bounds = [int(means[0][region][0] - x_diff), int(means[0][region][0] + x_diff)]
        y_bounds = [int(means[0][region][1] - y_diff), int(means[0][region][1] + y_diff)]

        if x_bounds[0] < 0:
            x_bounds[0] = 0

        if x_bounds[0] >= size[0]:
            x_bounds[0] = size[0] - 1;

        if y_bounds[0] < 0:
            y_bounds[0] = 0

        if y_bounds[1] >= size[1]:
            y_bounds[1] = size[1] - 1;

        
        current_box = [
            (x_bounds[0], y_bounds[0]),
            (x_bounds[0], y_bounds[1]),
            (x_bounds[1], y_bounds[0]),
            (x_bounds[1], y_bounds[1]),
        ]

        if j == 0:
            initial_boxes.append(current_box)
            region_indices.append(descending_distances_indices[j])
        else:
            current_box_overlaps = False
            for prev_box in initial_boxes:
                if not current_box_overlaps \
                    and (does_overlap(current_box, prev_box) \
                         or does_overlap(prev_box, current_box)):
                        current_box_overlaps = True

            if not current_box_overlaps:
                initial_boxes.append(current_box)
                region_indices.append(descending_distances_indices[j])

        j += 1

    return region_indices

def spatial_anomaly_detection(frames, means, covars, eigen_vecs, 
                        k, fps, size, outdir_path, std_threshold=3):
    '''
    Draws bounding boxes around the mixture component
    regions demonstrating the most variance.

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
        Number of the most significant non-overlapping 
        regions to display bounding boxes for.
    fps: int
        Frames per second of the video.
    size: tuple (2,)
        Width and height of the video.
    outdir_path: string
        Path to save the bounding box video.
    std_threshold: float 
        The number of standard deviations to use to compute
        the spatial region of the bounding box. Default is
        three.
    '''

    out_vid_path = os.path.join(outdir_path, 'bounding_box_example.mp4')
    initial_boxes = []
    region_indicies = []
    box_colors = {}
    for i in range(k):
        box_colors[i] = np.random.randint(256, size=(3,))
        #box_colors[i] = (30, 144, 255)
    
    distances = absolute_distance_traveled(eigen_vecs)   
    descending_distances_indices = np.flip(np.argsort(distances))

    region_indices = find_initial_boxes(means, covars, size, 
                                        descending_distances_indices, k)
    with imageio.get_writer(out_vid_path, mode='I', fps=1) as writer:
        for i, frame in enumerate(tqdm(frames)):
                for index, j in enumerate(region_indices):
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
                        
                    color = box_colors[index]
                    frames[i][x_bounds[0]:x_bounds[1], y_bounds[0], :] = color
                    frames[i][x_bounds[0]:x_bounds[1], y_bounds[1], :] = color
                    frames[i][x_bounds[0], y_bounds[0]:y_bounds[1], :] = color
                    frames[i][x_bounds[1], y_bounds[0]:y_bounds[1], :] = color
                
                writer.append_data(frames[i])

def parse_cli(args):
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
        description='Indicates spatial regions demonstrating the relatively' + 
                    ' high variance with bounding boxes.'
    )
    parser.add_argument('-i', '--intermediates', required=True,
                         help='GMM intermediates file (.npz).')
    parser.add_argument('-e', '--eigendata', required=True,
                         help='Eigendata file (.npz).')
    parser.add_argument('-v', '--video', required=True,
                         help='Individual cell video path (.avi).')
    parser.add_argument('-o', '--outdir', default=os.getcwd(),
                         help='Output directory path.')
    parser.add_argument('-k', '--box_number', default=1, type=int,
                         help='Number of bounding boxes to be utilized.')

    return vars(parser.parse_args(args))

def main():
    args = parse_cli(sys.argv[1:])
    eigen_data = np.load(args['eigendata'])
    eigen_vals, eigen_vecs = eigen_data['eigen_vals'], eigen_data['eigen_vecs']
    inter = np.load(args['intermediates'])
    means, covars = inter['means'], inter['covars']
    with imageio.get_reader(args['video']) as reader:
        frames = list(reader)
        fps = reader.get_meta_data()['fps']
        size = reader.get_meta_data()['size']

    spatial_anomaly_detection(frames, means, covars, eigen_vecs, 
                        args['box_number'], fps, size, args['outdir'])

if __name__ == '__main__':
    main()
