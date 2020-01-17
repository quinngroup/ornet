'''
Applies median normalization to an input video, and saves the output (.avi).'
'''
import os
import argparse

import cv2
import imageio
import numpy as np
from tqdm import tqdm


def median_normalize(vid_name, vid_path, out_path):
    '''
    Parameters
    ----------
    vid_name: String
        name for the output video
    vid_path: String
        path to the input video
    out_path: String
        path to the directory to save the ouptut video

    Returns
    ----------
    NoneType object
    '''
    medians = []
    reader = imageio.get_reader(vid_path)
    progress_bar = tqdm(total=(2 * reader.count_frames()))
    progress_bar.set_description(' Normalizing video')
    for frame in reader:
        grayscale_frame = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        flat_frame = grayscale_frame.flatten()
        flat_frame[flat_frame > 0]
        medians.append(np.median(flat_frame))
        progress_bar.update()

    medians = np.array(medians, dtype=np.uint8)
    max_median = np.max(medians)
    adjusted_medians = medians - max_median

    reader = imageio.get_reader(vid_path)
    output = os.path.join(out_path, vid_name + '.avi')
    fps = reader.get_meta_data()['fps']
    size = reader.get_meta_data()['size']
    writer = cv2.VideoWriter(output, 
             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
             fps, size)

    for i, frame in enumerate(reader):
        grayscale_frame = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        flat_frame = grayscale_frame.flatten()
        flat_frame[flat_frame != 0] += adjusted_medians[i]
        out_frame = np.array(flat_frame, dtype=np.uint8).reshape(size)
        color_frame = cv2.cvtColor(out_frame, cv2.COLOR_GRAY2RGB)
        writer.write(color_frame)
        progress_bar.update()
    
    progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies median \
                                                  normalization to an input \
                                                  video, and saves the output \
                                                  (.avi)')
    parser.add_argument('-i', '--input', required=True,
                        help='Input video path')
    parser.add_argument('-o', '--output', required=True,
                        help='Output video path')
    args = vars(parser.parse_args())

    vid_name = os.path.split(args['input'])[-1].split('.')[0] + '_normalized'
    median_normalize(vid_name, args['input'], args['output'])
