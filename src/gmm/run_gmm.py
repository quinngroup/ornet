import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from sklearn.mixture import GaussianMixture

from ornet.gmm import image, params, viz


def skl_gmm(vid, vizual=False, skipframes=1, threshold_abs=6, min_distance=10):
    """
    Runs a warm-start GMM over evenly-spaced frames of the video.

    Parameters
    ----------
    vid : array, shape (f, x, y)
        Video, with f frames and spatial dimensions x by y.
    vizual : boolean
        True will show images and nodes (default: False).
    skipframes : integer
        Number of frames to skip (downsampling constant).
    threshold_abs: int
        Absolute minimum pixel value to be used in 
        scikit-image's peak_local max function
    min_distance: int
        Minimum distance between image peaks that will be 
        returned by scikit-image's peak_local max function

    Returns
    -------
    means : array, shape (f, k, 2)
        The k 2D means for each of f frames.
    covars : array, shape (f, k, 2, 2)
        The k covariance matrices (each 2x2) for each of f frames.
    weights : array, shape (f, k)
        The k weights for each of f frames.
    precisions : array, shape (f, k, 2, 2)
        The k precision matrices for each of f frames.
    """
    img = vid[0]
    if vizual:
        plt.imshow(img)
        plt.show()
    X = image.img_to_px(img)
    PI, MU, CV = params.image_init(img, k=None,
                                   min_distance=min_distance,
                                   threshold_abs=threshold_abs)

    PR = np.array(list(map(sla.inv, CV)))
    gmmodel = GaussianMixture(n_components=CV.shape[0], weights_init=PI,
                              means_init=MU, precisions_init=PR)
    gmmodel.fit(X)
    if vizual:
        viz.plot_results(gmmodel.means_, gmmodel.covariances_,
                         0, img.shape[1], 0, img.shape[0], 0, 'this')

    covars = [gmmodel.covariances_]
    means = [gmmodel.means_]
    weights = [gmmodel.weights_]
    precisions = [gmmodel.precisions_]

    # set warm start to true to use previous parameters
    gmmodel.warm_start = True

    for i in range(0 + skipframes, vid.shape[0], skipframes):
        img = vid[i]
        if vizual:
            plt.imshow(img)
            plt.show()

        X = image.img_to_px(img)
        gmmodel.fit(X)
        covars = np.append(covars, [gmmodel.covariances_], axis=0)
        means = np.append(means, [gmmodel.means_], axis=0)
        weights = np.append(weights, [gmmodel.weights_], axis=0)
        precisions = np.append(precisions, [gmmodel.precisions_], axis=0)

        if vizual:
            viz.plot_results(gmmodel.means_, gmmodel.covariances_,
                             0, img.shape[1], 0, img.shape[0], 0, 'this')

    return means, covars, weights, precisions


def run_gmm(vid, vizual=False, skipframes=1, threshold_abs=6, min_distance=10):
    """
    Runs packaged GMM reimplementation over evenly-spaced frames of the video.

    Parameters
    ----------
    vid : array, shape (f, x, y)
        Video, with f frames and spatial dimensions x by y.
    vizual : boolean
        True will show images and nodes (default: False).
    skipframes : integer
        Number of frames to skip (downsampling constant).
    threshold_abs: int
        Absolute minimum pixel value to be used in 
        scikit-image's peak_local max function
    min_distance: int
        Minimum distance between image peaks that will be 
        returned by scikit-image's peak_local max function

    Returns
    -------
    MU : array, shape (f, k, 2)
        The k 2D means for each of f frames.
    CV : array, shape (f, k, 2, 2)
        The k covariance matrices (each 2x2) for each of f frames.
    PI : array, shape (f, k)
        The mixing coefficients for each frame.
    """
    return
