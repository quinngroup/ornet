'''
Draws bounding boxes around subcellular demonstrating simalar
anomalous behavior.
'''

import os
import sys
import csv
import math
import argparse

import imageio
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

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

def compute_region_boundaries(means, covars, size, frame, region, std_threshold=3):
    '''
    Computes the boundaries of a box for the given region on a specific frame.

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
    frame: int
        The index of the video frame to utilize.
    region: int
        The index of the spatial region to create a box for.
    std_threshold: float 
        The number of standard deviations to use to compute
        the spatial region of the bounding box. Default is
        three.

    Returns
    -------
    row_bounds, col_bounds: list (2,)
        The row and column boundaries of the bounding box.
        For row_bounds index 0 contains the row value 
        corresponding to the top of the bounding box, while 
        index 1 contains the row value corresponding to the
        bottom of the box; Likewise, index 0 of col_bounds
        corresponds to the left column value of the box and
        index 1 corresponds to the right column value of the
        box.
    '''

    row_diff = std_threshold * math.sqrt(covars[frame][region][0][0])
    col_diff = std_threshold * math.sqrt(covars[frame][region][1][1])
    row_bounds = [
        int(means[frame][region][0] - row_diff), 
        int(means[frame][region][0] + row_diff)
    ]
    col_bounds = [
        int(means[frame][region][1] - col_diff), 
        int(means[frame][region][1] + col_diff)
    ]

    if row_bounds[0] < 0:
        row_bounds[0] = 0

    if row_bounds[1] >= size[0]:
        row_bounds[1] = size[0] - 1;

    if col_bounds[0] < 0:
        col_bounds[0] = 0

    if col_bounds[1] >= size[1]:
        col_bounds[1] = size[1] - 1;

    return row_bounds, col_bounds

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
    if box_one[0][0] >= box_two[0][0] and box_one[0][0] <= box_two[2][0] \
        and box_one[0][1] >= box_two[0][1] and box_one[0][1] <= box_two[1][1]:
        overlap_status = True
    elif box_one[2][0] >= box_two[0][0] and box_one[2][0] <= box_two[2][0] \
        and box_one[2][1] >= box_two[0][1] and box_one[2][1] <= box_two[1][1]:
        overlap_status = True
    elif box_one[1][0] >= box_two[0][0] and box_one[1][0] <= box_two[2][0] \
        and box_one[1][1] >= box_two[0][1] and box_one[1][1] <= box_two[1][1]:
        overlap_status = True
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
        row_bounds, col_bounds = compute_region_boundaries(means, covars, 
                                                           size, 0, region)
        current_box = [
            (row_bounds[0], col_bounds[0]),
            (row_bounds[0], col_bounds[1]),
            (row_bounds[1], col_bounds[0]),
            (row_bounds[1], col_bounds[1]),
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

def spatial_anomaly_detection(vid_path, means, covars, eigen_vecs, 
                        k, outdir_path, std_threshold=3, display_areas=False):
    '''
    Draws bounding boxes around the mixture component
    regions demonstrating the most variance.

    Parameters
    ----------
    vid_path: string
        Path to the input video.
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
        regions to display bounding boxes for. The
        actual number may be less than k, if the video
        does not contain that many non-overlapping 
        regions.
    outdir_path: string
        Directory path to save the bounding box video.
    std_threshold: float 
        The number of standard deviations to use to compute
        the spatial region of the bounding box. Default is
        three.
    display_areas: bool
       Indicates whether to print the areas of the bounding
       boxes.

    Returns
    -------
    NoneType object
    '''

    out_vid_path = os.path.join(
        outdir_path, os.path.split(vid_path)[1].split('.')[0] + '.mp4'
    )
    with imageio.get_reader(vid_path) as reader, \
         imageio.get_writer(out_vid_path, mode='I', fps=100) as writer:
        fps = reader.get_meta_data()['fps']
        size = reader.get_meta_data()['size']
        distances = absolute_distance_traveled(eigen_vecs)   
        descending_distances_indices = np.flip(np.argsort(distances))
        region_indices = find_initial_boxes(means, covars, size, 
                                        descending_distances_indices, k)
        num_of_boxes = len(region_indices)
        box_colors = np.random.randint(256, size=(num_of_boxes, 3))

        for i, frame in enumerate(reader):
            current_frame = frame
            avg_box_area = 0
            if display_areas:
                print('Frame', i)

            for index, j in enumerate(region_indices):
                row_bounds, col_bounds = compute_region_boundaries(
                    means, covars, size, i, j
                )
                row_diff = row_bounds[1] - row_bounds[0]
                col_diff = col_bounds[1] - col_bounds[0]
                current_box_area = row_diff * col_diff
                avg_box_area = current_box_area / num_of_boxes
                if display_areas:
                    print('Box ', index, 'Area: ', current_box_area)

                current_frame[row_bounds[0]:row_bounds[1], col_bounds[0], :] = box_colors[index]
                current_frame[row_bounds[0]:row_bounds[1], col_bounds[1], :] = box_colors[index]
                current_frame[row_bounds[0], col_bounds[0]:col_bounds[1], :] = box_colors[index]
                current_frame[row_bounds[1], col_bounds[0]:col_bounds[1], :] = box_colors[index]
            
            writer.append_data(current_frame)
            if display_areas:
                print('Average box area: ', avg_box_area, '\n')

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
    spatial_anomaly_detection(args['video'], means, covars, eigen_vecs, 
                        args['box_number'], args['outdir'])

if __name__ == '__main__':
    main()
