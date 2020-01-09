import argparse
from functools import partial
import os
import joblib

import numpy as np

from ornet.gmm.run_gmm import skl_gmm

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description=('Reading all npy files in directory, passing through ',
                     'gaussian mixture model and saving intermediates'),
        add_help='How to use', prog='make_gmm_intermediates.py <args>')

    # Required arguments.
    parser.add_argument("-i", "--input", required=True,
                        help=("Path to a single npy file or directory of "
                              "npy files."))

    # Optional arguments.
    parser.add_argument("-o", "--output", default=os.path.join(cwd, "videos"),
                        help=("Destination path for intermediates."
                              " [DEFAULT: cwd]"))
    parser.add_argument("-s", "--skipframes", default=1,
                        help=("Number of frames to skip (downsample) when"
                              " reading videos. [DEFAULT: 1]"))
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help=("Degree of parallelism for reading in videos."
                              " -1 is all cores. [DEFAULT -1]"))

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    if os.path.isdir(args['input']):
        files = filter(lambda x: x.split(".")[-1] in ["npy"],
                       os.listdir(args['input']))
        prefix = partial(os.path.join, args['input'])
        vidpaths = list(map(prefix, files))
    else:
        vidpaths = [args['input']]

    # Spawn parallel jobs to read the videos in the directory listing.
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10)(
        joblib.delayed(skl_gmm)
        (np.load(v), vizual=False, skipframes=args['skipframes'])
        for v in vidpaths
    )

    # Write the files out.
    for outs, v in zip(out, vidpaths):
        key = v.split(os.path.sep)[-1].split(".")[0]
        print(key)
        fname = "{}_intermediates.npz".format(key)
        outfile = os.path.join(args['output'], fname)
        np.savez(outfile, means=outs[0], covars=outs[1],
                 weights=outs[2], precs=outs[3])
