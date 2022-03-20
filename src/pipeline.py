'''
A script that passes input video(s) through the entire OrNet pipeline. 
The pipeline consists of cell segmentation, graph vertex construction 
via Gaussian mixture model means, edge construction via divergence 
functions, and eigen-decomposition of the matrix representation.
'''

import os
import re
import shutil

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from ornet.gmm.run_gmm import skl_gmm
from ornet.cells_to_gray import vid_to_gray
from ornet.affinityfunc import get_all_aff_tables
from ornet.extract_cells import extract_cells
from ornet.median_normalization import median_normalize as normalize

def constrain_vid(vid_path, out_path, constrain_count, display_progress=True):
    '''
    Constrains the input video to specified number of frames, and write the
    result to an output video (.avi). If the video contains less frames than
    constrain_count, then then all frames of the video are returned.

    Parameters
    ----------
    vid_path: String
        Path to the input video.
    out_path: String
        Path to the output video.
    constrain_count: int
        First N number of frames to extract from the video.
        If value is -1, then the entire video is used.
    display_progress: bool
        Flag that indicates whether to show the progress bar
        or not.

    Returns
    ----------
    NoneType object
    '''

    with imageio.get_reader(vid_path) as reader:
        fps = reader.get_meta_data()['fps']
        size = reader.get_meta_data()['size']
        count = reader.count_frames()

        if constrain_count == -1:
            constrain_count = count 

        with imageio.get_writer(out_path, mode='I', fps=fps) as writer:
            if display_progress:
                progress_bar = tqdm(total=constrain_count)
                progress_bar.set_description('Constraining video')

            i = 0
            for frame in reader:
                if i == constrain_count:
                    break
                else:
                    writer.append_data(frame)
                    i += 1

                if display_progress:
                    progress_bar.update()

            if display_progress:
                progress_bar.close()
            else:
                print("Video constraining complete.")

def median_normalize(vid_name, input_path, out_path):
    '''
    Applies median normalization to a grayscale input video (.npy)

    Parameters
    ----------
    vid_name: String
        Name of the input video.
    input_path: String
        Path to the grayscale video (.npy file).
    out_path: String
        Directory to save the normalized video.

    Returns
    ----------
    NoneType object
    '''
    normalize(vid_name, input_path, out_path)


def gray_to_avi(vid_name, gray_path, original_path, out_path):
    '''
    Converts a grayscale representation of a video (numpy file) into an
    .avi video.

    Parameters
    ----------
    vid_name: String
        Name of the input video
    gray_path: String
        Path to the numpy file (.npy)
    original_path: String
        Path to the original video
    out_path: String
        Path to save the output video

    Returns
    ----------
    NoneType object
    '''

    reader = imageio.get_reader(original_path)
    fps = reader.get_meta_data()['fps']
    size = reader.get_meta_data()['size']
    reader.close()

    gray_vid = np.load(gray_path)
    writer = cv2.VideoWriter(os.path.join(out_path, vid_name + '.avi'),
                             cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'), fps,
                             size)
    for frame in gray_vid:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    writer.release()


def downsample_vid(vid_name, vid_path, out_dir_path, frame_skip):
    '''
    Takes an input video and saves a downsampled version 
    of it, by skipping a specified number of frames. The
    saved video is (.avi) format.

    Parameters
    ----------
    vid_name: String
        Name of the input video.
    vid_path: String
        Path to the input video.
    out_dir_path:
        Path to directory where the downsampled video will be saved.
    frame_skip:
        The number of frames to skip for downsampling.

    Returns
    ----------
    NoneType object
    '''
    with imageio.get_reader(vid_path) as reader:
        fps = reader.get_meta_data()['fps']
        size = reader.get_meta_data()['size']
        count = reader.count_frames()
        output_path = os.path.join(out_dir_path, vid_name + '.avi')
        with imageio.get_writer(output_path ,mode='I', fps=fps) as writer:
            progress_bar  = tqdm(total=count)
            progress_bar.set_description('      Downsampling')
            for i, frame in enumerate(reader):
                if i % frame_skip == 0:
                    writer.append_data(frame)

                progress_bar.update()

            progress_bar.close()

def generate_single_vids(vid_path, masks_path, output_path, downsample=1):
    '''
    Extracts individual cells using the segmentation masks.

    Parameters
    ----------
    vid_path: String
        Path to input video.
    masks_path: String
        Path to the segmentation mask for the input video.
    output_path: String
        Directory to save the individual videos.
    downsample: int
        The number of frames to skip. Default is 1, thus no
        downsampling.

    Returns
    ----------
    NoneType object
    '''
    extract_cells(vid_path, masks_path, output_path, downsample)


def convert_to_grayscale(vid_path, output_path):
    '''
    Converts an input video into an array of grayscale frames

    Parameters
    ----------
    vid_path: String
        Path to a single video.
    output_path: String
        Directory to save the grayscale frames.

    Returns
    ----------
    NoneType object
    '''

    vid_to_gray(vid_path, output_path, False)


def compute_gmm_intermediates(vid_dir, intermediates_path):
    '''
    Generate intermediate files from passing a grayscale video
    through the GMM portion of the pipeline.

    Parameters
    ----------
    vid_dir: String
        Path to the directory that contains the single videos.
    intermediates_path:
        Path to save the intermediate files.

    Returns
    ----------
    NoneType object
    '''

    file_names = os.listdir(vid_dir)
    gray_vids = [x for x in file_names if x.split('.')[-1] in ['npy']]

    progress_bar = tqdm(total=len(gray_vids))
    progress_bar.set_description('Computing GMM info')
    for vid_name in gray_vids:
        try:
            vid_path = os.path.join(vid_dir, vid_name)
            vid = np.load(vid_path)

            means, covars, weights, precisions, predictions = skl_gmm(vid)
            np.savez_compressed(os.path.join(intermediates_path,
                                  vid_name.split('.')[0] + '.npz'),
                     means=means, covars=covars, weights=weights,
                     precs=precisions, preds=predictions)
        except:
            print('Disappering cell: ' + vid_name)

        progress_bar.update()

    progress_bar.close()


def compute_distances(intermediates_path, distances_path):
    '''
    Generate distances between means using Hellinger Distance.

    Parameters
    ----------
    intermediates_path: String
        Path to the GMM intermediates.
    distances_path: String
        Directory to save the distance ouptuts.

    Returns
    ----------
    NoneType object
    '''

    all_intermediates = os.listdir(intermediates_path)
    distances = os.listdir(distances_path)
    intermediates = []
    for x in all_intermediates:
        file_name = x.split('.')[0]
        if file_name + '.npy' not in distances:
           intermediates.append(file_name)
    
    progress_bar = tqdm(total=len(intermediates))
    progress_bar.set_description('Computing distance')
    for intermediate in intermediates:
        vid_inter = np.load(os.path.join(intermediates_path, intermediate + '.npz'))
        table = get_all_aff_tables(vid_inter['means'], vid_inter['covars'],
                                   'Hellinger')
        np.save(os.path.join(distances_path, intermediate + '.npy'), table)
        progress_bar.update()

    progress_bar.close()

def run(input_path, initial_masks_dir, output_path, constrain_count=-1, 
        downsample=1):
    '''
    Runs the entire ornet pipeline from start to finish for any video(s)
    found at the input path location.

    Paramaters
    ----------
    input_path: String
        Path to input video(s).
    initial_masks_dir: String
        Path to the directory contatining the initial 
        segmentation mask that corresponds with the input 
        video.
    output_path: String
        Path to the output directory.
    constrain_count: int
        The first N number of frames of the video to use.
    downsample: int
        The number of frames to skip when performing
        downsampling.

    Returns
    ----------
    NoneType object
    '''

    if os.path.isdir(input_path):
        vids = [x for x in os.listdir(input_path) if
                x.split('.')[-1] in ['avi', 'mov', 'mp4']]
        input_dir = input_path
    else:
        input_dir, vids, = os.path.split(input_path)
        if vids.split('.')[-1] in ['avi', 'mov', 'mp4']:
            vids = [vids]
        else:
            vids = []
    num_of_vids = len(vids)
    if num_of_vids == 0:
        print('No videos were found.')
        quit(1)

    for i, vid in enumerate(vids):
        #print(str(i) + ':', vid)
        print(vid + ':', '(' + str(i + 1) + '/' + str(num_of_vids) + ')')
        out_path = os.path.join(output_path, 'ornet-outputs')
        vid_name = vid.split('.')[0]
        vid_name = re.sub(' \(2\)| \(Converted\)', '', vid_name)
        full_vid_path = os.path.join(out_path, vid_name + '.avi')
        mask_path = os.path.join(initial_masks_dir, vid_name + '.vtk')
        normalized_dir_path = os.path.join(out_path, 'normalized')
        normalized_vid_path = os.path.join(normalized_dir_path, 
                                           vid_name + '_normalized.avi')
        singles_dir_path = os.path.join(out_path, 'singles')
        intermediates_dir_path = os.path.join(out_path, 'intermediates')
        distances_dir_path = os.path.join(out_path, 'distances')
        tmp_dir_path = os.path.join(out_path, 'tmp')

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(normalized_dir_path, exist_ok=True)
        os.makedirs(singles_dir_path, exist_ok=True)
        os.makedirs(intermediates_dir_path, exist_ok=True)
        os.makedirs(distances_dir_path, exist_ok=True)
        os.makedirs(tmp_dir_path, exist_ok=True)

        constrain_vid(os.path.join(input_dir, vid), full_vid_path, 
                      constrain_count)
        median_normalize(vid_name + '_normalized', full_vid_path, 
                         normalized_dir_path)
        generate_single_vids(full_vid_path, mask_path, singles_dir_path, 
                             downsample)
        generate_single_vids(normalized_vid_path, mask_path, tmp_dir_path, 
                             downsample)

        single_vids = os.listdir(tmp_dir_path)
        progress_bar = tqdm(total=len(single_vids))
        progress_bar.set_description('Converting to gray')
        for single in single_vids:
            convert_to_grayscale(os.path.join(tmp_dir_path, single), 
                                 tmp_dir_path)
            shutil.move(os.path.join(tmp_dir_path, single),
                        os.path.join(singles_dir_path, single))
            progress_bar.update()

        progress_bar.close()

        compute_gmm_intermediates(tmp_dir_path, intermediates_dir_path)
        compute_distances(intermediates_dir_path, distances_dir_path)

        os.remove(full_vid_path)
        shutil.rmtree(normalized_dir_path)
        shutil.rmtree(tmp_dir_path)
        print() #New line for output formatting
