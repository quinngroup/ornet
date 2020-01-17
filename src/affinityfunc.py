import os
import argparse
from functools import partial

import joblib
import numpy as np

from ornet.gmm.loss import normpdf
from ornet.measure import multivariate_js, multivariate_kl, \
    multivariate_hellinger


def aff_by_eval(means, covars):
    """
    finds an affinity table for a set of
    means and covariances representing nodes in a graph

    Parameters
    ----------
    means : array, shape (k, 2)
        the list of means with k nodes
    covars : array, shape (k, 2, 2)
        the list of covars with k nodes

    Returns
    -------
    aff_Table : array, shape (k, k)

    """
    aff_Table = np.empty([means.shape[0], 0])
    # iterate through the components
    for i, (mean, covar) in enumerate(zip(means, covars)):
        # find the probability for every mean point in the current component
        p_mus_Kx = normpdf(means, mean, covar)
        # transpose the probabilities and append them as a column
        aff_Table = np.append(aff_Table, np.transpose([p_mus_Kx]), axis=1)
    return aff_Table


def aff_KL_div(means, covars):
    """
    Applies KL divergence to each pair of intermediates to create an affinity
    table for a frame
    """

    aff_Table = np.empty([means.shape[0], means.shape[0]])
    # iterate through the components
    for i, (mean_1, covar_1) in enumerate(zip(means, covars)):
        # find the probability for every mean point in the current component
        for j, (mean_2, covar_2) in enumerate(zip(means, covars)):
            aff_Table[i, j] = multivariate_kl(mean_1, covar_1, mean_2, covar_2)
        # transpose the probabilities and append them as a column
    return aff_Table


def aff_JS_div(means, covars):
    """
    Applies Jensen-Shannon divergence to each pair of intermediates to create an affinity
    table for a frame
    """

    aff_Table = np.empty([means.shape[0], means.shape[0]])
    # iterate through the components
    for i, (mean_1, covar_1) in enumerate(zip(means, covars)):
        # find the probability for every mean point in the current component
        for j, (mean_2, covar_2) in enumerate(zip(means, covars)):
            aff_Table[i, j] = multivariate_js(mean_1, covar_1, mean_2, covar_2)
        # transpose the probabilities and append them as a column
    return aff_Table


def aff_hellinger(means, covars):
    """
    Applies Jensen-Shannon divergence to each pair of intermediates to create an affinity
    table for a frame
    """

    aff_Table = np.empty([means.shape[0], means.shape[0]])
    # iterate through the components
    for i, (mean_1, covar_1) in enumerate(zip(means, covars)):
        # find the probability for every mean point in the current component
        for j, (mean_2, covar_2) in enumerate(zip(means, covars)):
            aff_Table[i, j] = multivariate_hellinger(mean_1, covar_1, mean_2,
                                                     covar_2)
        # transpose the probabilities and append them as a column
    return aff_Table


def get_all_aff_tables(means, covars, aff_funct, progress=True):
    """
    finds all affinity table for a set of Frames
    each with lists of means and covariances

    Parameters
    ----------
    means : array, shape (f, k, 2)
        the list of lists of means with f frames and k nodes
    covars : array, shape (k, 2, 2)
        the list of lists of covars with f frames with k nodes
    aff_funct: string
        the affinity metric that will be applied to the 
        distributions
    progress: bool
        flag to display a progress bar

    Returns
    -------
    aff_Table : array, shape (f, k, k)

    """

    aff_dispatch = {
        'probability': aff_by_eval,
        'KL div': aff_KL_div,
        'JS div': aff_JS_div,
        'Hellinger': aff_hellinger
    }
    aff_Tables = [aff_dispatch[aff_funct](means[0], covars[0])]
    for i in range(1, means.shape[0]):
        aff_Tables = np.append(
            aff_Tables, 
            [aff_dispatch[aff_funct](means[i], covars[i])],
            axis=0
        )

    return aff_Tables


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description=('Reads all npz files of intermediates in directory ',
                     'formulates affinity tables for each frame'),
        add_help='How to use', prog='affinityfunc.py <args>')

    # Required arguments.
    parser.add_argument("-i", "--input", required=True,
                        help=("Path to a single npz file or directory of "
                              "npz files."))

    # Optional arguments.
    parser.add_argument("-o", "--output", default=os.path.join(cwd, "videos"),
                        help=("Destination path for affinity tables."
                              " [DEFAULT: cwd]"))
    parser.add_argument("-a", "--affinity_type", default='probability',
                        help=("Degree of parallelism for reading in videos."
                              " 'probability' is A -> B = A(B).",
                              " 'KL div' is Kullback Leibler divergence",
                              " 'JS div' is Jenson Shannon divergence",
                              " 'Hellinger' is Hellinger distance"
                              "[DEFAULT probability]"))
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help=("Degree of parallelism for reading in videos."
                              " -1 is all cores. [DEFAULT -1]"))

    args = vars(parser.parse_args())
    print(args['output'])
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    if os.path.isdir(args['input']):
        files = filter(lambda x: x.split(".")[-1] in ["npz"],
                       os.listdir(args['input']))
        prefix = partial(os.path.join, args['input'])
        vidpaths = list(map(prefix, files))
    else:
        vidpaths = [args['input']]
    print(vidpaths)
    # Spawn parallel jobs to read the videos in the directory listing.
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10)(
        joblib.delayed(get_all_aff_tables)
        (np.load(v)['means'], np.load(v)['covars'], args['affinity_type'])
        for v in vidpaths
    )

    # Write the files out.
    for outs, v in zip(out, vidpaths):
        key = v.split(os.path.sep)[-1].split(".")[0]
        print(key)
        fname = "{}_aff_table.npy".format(key)
        outfile = os.path.join(args['output'], fname)
        np.save(outfile, outs)
