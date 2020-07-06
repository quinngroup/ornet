'''
Tracks segmentaiton masks, extracts the individual cells, 
and downsamples the individual cell videos.
'''

import os
import random
import argparse
from functools import partial

import cv2
import imageio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def extract_cells(vidfile, maskfile, output_path, downsample=1, show_video=False, display_progress=True):
    """
    Extracts the cells present in a microscopy video
    into indiviaul videos.

    Parameters
    ----------
    vidfile : string
        path to a single video file
    maskfile : string
        path to a single vtk mask file of the
        first frame.
    output_path: String
        Path to the directory to save the individual videos.
    downsample: int
        The number of frames to skip. Default is 1, thus no
        downsampling.
    show_video : boolean (Default : False)
        If true, display video with contours drawn during processing
    display_progress: bool
        Flag that indicates whether to show the progress bar
        or not.

    Returns
    ---------
    NoneType
    """

    writers = []
    vid_name = os.path.split(vidfile)[1].split('.')[0]
    os.makedirs(output_path, exist_ok=True)

    im = imageio.imread(maskfile)
    if show_video:
        plt.imshow(im)
        plt.show()

    vf = imageio.get_reader(vidfile)
    frameNum = 0
    number_of_segments = len(
        np.unique(im)) - 1  # defines number of segs from vtk

    for i in range(number_of_segments):
        vid_path = os.path.join(
            output_path, 
            vid_name + '_' + str(i + 1) + '.avi'
        )
        writers.append(imageio.get_writer(vid_path, mode='I', fps=100)) #fps

    outs = []
    ims = list()
    masks = list()
    colors = list()  # the colors to make the contours in the output video?
    contours = list()  # holds contours
    dilates = list()
    comparisons = list()

    kernel = np.ones((17, 17), np.uint8)  # kernel for opening
    kernel2 = np.ones((3, 3), np.uint8)  # kernel for dilation
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for frame count
        
    for cols in range(number_of_segments):  # creates random colors to use for the outlines
        colors.append((random.randint(1, 255), random.randint(0, 255),
                       random.randint(1, 255)))

    for i in range(number_of_segments):  # separates each mask from the vtk and lists them
        masks.append(im != i + 1)

    if display_progress:
        progress_bar = tqdm(total=vf.count_frames())
        progress_bar.set_description('    Extracting cells')

    for frameNum, frame in enumerate(vf):  # while( vf.isOpened() ):
        original_frame = frame.copy()
        for i in range(number_of_segments):  # adds a copy of the current frame for each segment
            ims.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        for i in range(number_of_segments):  # blacks out all that isn't in the initial mask for each segment
            if (frameNum == 0):
                ims[i][masks[i]] = 0
            else:
                ims[i][dilates[
                           i] != 255] = 0  # uses previous dilation for next frame

        del dilates[:]
        for i in range(number_of_segments):
            ret, ims[i] = cv2.threshold(ims[i], 3, 255, cv2.THRESH_BINARY_INV)
            temp = cv2.morphologyEx(ims[i], cv2.MORPH_OPEN, kernel,
                                    iterations=3)
            temp = cv2.bitwise_not(temp)
            dilates.append(temp)

        # each segment dilates
        # if two segments overlap cut the overlapping portions from both (first one to cover a place gets it...)
        if (
                frameNum % 100 == 1 or frameNum < 20):  # every 100 frames do full pairwise comparisons and save which were adjacent
            for i in range(5):
                del comparisons[:]
                for d in range(number_of_segments):
                    dilates[d] = cv2.dilate(dilates[d], kernel2, iterations=2)
                for s in range(
                        number_of_segments):  # tests each segment with every other segment, slow
                    newcomps = list()
                    for t in range(s + 1, number_of_segments):
                        b_and = cv2.bitwise_and(dilates[s], dilates[t])
                        if (len(np.unique(b_and)) > 1):
                            newcomps.append(t)
                            dilates[s] = cv2.bitwise_xor(dilates[s], b_and)
                            dilates[t] = cv2.bitwise_xor(dilates[t], b_and)
                    comparisons.append(newcomps)

        else:  # every other frame only compare those that overlapped previously
            for i in range(5):
                for d in range(number_of_segments):
                    dilates[d] = cv2.dilate(dilates[d], kernel2, iterations=2)
                for s in range(number_of_segments):
                    for t in comparisons[s]:
                        b_and = cv2.bitwise_and(dilates[s], dilates[t])
                        if (len(np.unique(b_and)) > 1):
                            dilates[s] = cv2.bitwise_xor(dilates[s], b_and)
                            dilates[t] = cv2.bitwise_xor(dilates[t], b_and)

        for conts in range(number_of_segments):  # bulds the contours and draws them onto the frame
            contours.append(cv2.findContours(dilates[conts], cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)[0])
            cv2.drawContours(frame, contours[conts], -1, colors[conts], 2)
            if (conts == number_of_segments - 1 and show_video):  # prints contours frame by frame
                cv2.putText(frame, 'Frame # ' + str(frameNum), (10, 40), font,
                            0.5, (0, 255, 50), 1)
                cv2.imshow("Keypoints2", frame)
                k = cv2.waitKey(10)
                if k == 27:
                    break

        frameMask = np.zeros_like(dilates[0])
        for i in range(number_of_segments):
            frameMask[dilates[i] == 255] = i + 1

        if frameNum % downsample == 0:
            for j in range(number_of_segments):
                mask = cv2.cvtColor(frameMask, cv2.COLOR_GRAY2RGB)
                output = np.ma.masked_where(mask != j + 1, original_frame)
                output = np.ma.filled(output, 0)
                writers[j].append_data(output)

        del ims[:]
        del contours[:]
        del masks[:]
        if display_progress:
            progress_bar.update()

    vf.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for writer in writers:
        writer.close()

    if display_progress:
        progress_bar.close()

def main():
    extract_cells('/home/marcus/Desktop/Scipy/Scipy-2020/samples/llo/llo.mp4',
                '/home/marcus/Desktop/Scipy/Scipy-2020/samples/llo/llo.vtk',
                '/home/marcus/Desktop/singles-test',
                100,
                show_video=True
    )
        
if __name__ == '__main__':
    main()
