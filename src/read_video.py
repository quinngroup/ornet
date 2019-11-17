'''
Frame Extractor : Mojtaba 09.06.2017
This script will receive a directory (which contains the video files), a
number of the frames that should be extracted from each video, and finally a
destination for saving the frames (default is the current root) as arguments.
It then extracts the first-r frames of each video that could be in one
of the *.mov, *.avi or *.mpg formats.
'''
import argparse
from functools import partial
import os

import imageio
import joblib
import numpy as np
import skimage
import skimage.color

def read_video(vidfile, read_every = 100, first_only = False, max_frames = -1, format = "ffmpeg"):
    """
    Reads a video file, and returns a NumPy array version of it.

    Parameters
    ----------
    vidfile : string
        Path to a single video file.
    read_every : integer (optional: 100)
        Downsamples a video by only taking every frame in a certain number.
    first_only : boolean (default: False)
        If True, returns only the first frame. Supersedes max_frames.
    max_frames : integer (defaut: -1)
        Maximum number of frames to retain. Superseded by first_only.
    format : string (default: "ffmpeg")
        Format argument for the imageio.get_reader object constructor.

    Returns
    -------
    name : string
        The name of the original file, as a sort of key for the caller.
    video : array, shape (H, W, F) or (H, W)
        Returns a uint8 grayscale NumPy array of the frame or frames.
    """
    vidcap = imageio.get_reader(vidfile, format = format, mode = "I")
    key = vidfile.split("/")[-1].split(".")[0]

    frames = []
    for index, frame in enumerate(vidcap):
        if index % read_every > 0: continue

        frame = _numpify(frame)
        if first_only:
            return [key, frame]
        frames.append(frame)
        if max_frames > -1 and len(frames) >= max_frames: break
    return [key, np.array(frames)]

def _numpify(frame):
    return skimage.img_as_ubyte(skimage.color.rgb2gray(frame))

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description = 'Reading frames of all videos in a specific directory and saving them as png files.',
        add_help = 'How to use', prog = 'read_video.py <args>')

    # Required arguments.
    parser.add_argument("-i", "--input", required = True,
        help = "Path to a single video file or directory of videos.")

    # Optional arguments.
    parser.add_argument("-o", "--output", default = os.path.join(cwd, "videos"),
        help = "Destination path for extracted frames. [DEFAULT: cwd]")
    parser.add_argument("-s", "--skipframes", default = 100,
        help = "Number of frames to skip (downsample) when reading videos. [DEFAULT: 100]")
    parser.add_argument("-m", "--maxframes", type = int, default = -1,
        help = "Total number of frames to be extracted for each video. -1 is all frames. [DEFAULT: -1]")
    parser.add_argument("-f", "--firstonly", action = "store_true",
        help = "If set, only the first frame of each video is saved. [DEFAULT: False]")
    parser.add_argument("--n_jobs", type = int, default = -1,
        help = "Degree of parallelism for reading in videos. -1 is all cores. [DEFAULT -1]")

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    if os.path.isdir(args['input']):
        files = filter(lambda x: x.split(".")[-1] in ["mpg", "avi", "mov"], os.listdir(args['input']))
        prefix = partial(os.path.join, args['input'])
        vidpaths = list(map(prefix, files))
    else:
        vidpaths = [args['input']]

    # Spawn parallel jobs to read the videos in the directory listing.
    out = joblib.Parallel(n_jobs = args['n_jobs'], verbose = 10)(
        joblib.delayed(read_video)
            (v, read_every = args['skipframes'], first_only = args['firstonly'], max_frames = args['maxframes'])
        for v in vidpaths
    )

    # Write the files out.
    for key, vid in out:
        if args['firstonly']:
            fname = "{}.png".format(key)
            outfile = os.path.join(args['output'], fname)
            imageio.imwrite(outfile, vid)
        else:
            fname = "{}.npy".format(key)
            outfile = os.path.join(args['output'], fname)
            np.save(outfile, vid)
