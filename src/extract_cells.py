'''
This script will take in both a video from a directory and an
initial masking image of the first frame. It will then create contours for
each subsequent frame and use those contours to mask out each cell. It will
then save each cell in their own video
'''
# Author : Andrew Durden

import os
import random
import argparse
from functools import partial

import cv2
import imageio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def extract_cells(vidfile, maskfile, show_video=False):
    """
    reads a video file and initial masks and returns a set of frames for each cell


    Parameters
    ----------
    vidfile : string
        path to a single video file
    maskfile : string
        path to a single vtk mask file
    show_video : boolean (Default : False)
        If true, display video with contours drawn during processing

    Returns
    ---------
    videos : Returns a list of arrays each with shape (H, W, F)
    """
    im = imageio.imread(maskfile)
    if show_video:
        plt.imshow(im)
        plt.show()

    vf = imageio.get_reader(vidfile)
    frameNum = 0
    number_of_segments = len(
        np.unique(im)) - 1  # defines number of segs from vtk
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

    progress_bar = tqdm(total=vf.count_frames())
    progress_bar.set_description('Tracking cells')
    for frameNum, frame in enumerate(vf):  # while( vf.isOpened() ):
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
        outs.append(frameMask)

        del ims[:]
        del contours[:]
        del masks[:]
        progress_bar.update()

    progress_bar.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return outs


if __name__ == '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description="Read Video File and VTK File and creates videos of each cell in the video",
        add_help="How to use", prog='extract_cells.py <args>')

    # Required args
    parser.add_argument("-i", "--input", required=True,
                        help="The path to a single video file")

    parser.add_argument("-m", "--masks", required=True,
                        help="The path to a single vtk file containing first frame masks")

    # Optional args
    parser.add_argument("-o", "--output",
                        default=os.path.join(cwd, "single_cell_videos"),
                        help="Destination path for output videos. [Default: cwd/single_cell_videos]")

    parser.add_argument("-s", "--showvid", action="store_true",
                        help="If set, each frame with contours is drawn during processing. [Default: False]")

    args = vars(parser.parse_args())

    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    vidpath = args['input']

    out = extract_cells(vidfile=args['input'], maskfile=args['masks'],
                        show_video=args['showvid'])
    fname = vidpath.split("/")[-1].split(".")[0]
    fname = "{}.npy".format(fname + 'MASKS')
    outfile = os.path.join(args['output'], fname)
    np.save(outfile, out)
