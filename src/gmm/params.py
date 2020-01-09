import numpy as np
import skimage.feature as feature


def random_init(X, k):
    """
    Provides an interface for randomly intializing the model parameters.

    Parameters
    ----------
    X : array, shape (N, d)
        The data.
    k : integer
        Number of components.

    Returns
    -------
    pi : array, shape (k,)
        List of initial mixing coefficients, set uniformly.
    means : array, shape (k, d)
        List of initial, randomized averages.
    covars : array, shape (k, d, d)
        List of initial, randomized covariances.
    """
    d = 1 if len(X.shape) == 1 else X.shape[1]
    pi = np.ones(k, dtype=np.float) / k
    if d == 1:
        means = (X.max() - X.min()) * np.random.random(k) + X.min()
        covars = np.abs((X.max() - X.min()) / 10) * np.random.random(k)
    else:
        # Pick random points in X for means.
        means = X[np.random.randint(0, high=X.shape[0], size=k)]

        # Covars is trickier; it's hard to get a quick sense of the scale of
        # d-dimensional data on-the-fly. Luckily, we can just iterate over
        # each dimension on its own and hope for the best.
        covars = np.zeros(shape=(k, d, d))
        for dimension in range(d):
            v = np.abs((X[:, dimension].max() - X[:,
                                                dimension].min()) / 10) * np.random.random(
                k)
            covars[:, dimension, dimension] = v

    # All done.
    return pi, means, covars


def image_init(image, k=None, min_distance=1, threshold_abs=None):
    """
    Initialization function for 2D histograms, i.e. images.

    Parameters
    ----------
    image : array, shape (H, W)
        The image, aka PDF, of the data.
    k : integer
        Number of components to return (default is all peaks from
        skimage.feature.peak_local_max).
    min_distance : integer
        See http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max.
    threshold_abs : integer
        See http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max.

    Returns
    -------
    pi : array, shape (k,)
        List of initial mixing coefficients, set as a function of pixel intensity.
    means : array, shape (k, 2)
        List of initial averages.
    covars : array, shape (k, 2, 2)
        List of initial covariances.
    """
    # Compute all the peaks.
    if k is not None:
        means = feature.peak_local_max(image, num_peaks=k,
                                       min_distance=min_distance,
                                       threshold_abs=threshold_abs)
    else:
        means = feature.peak_local_max(image, min_distance=min_distance,
                                       threshold_abs=threshold_abs)
    # Set up the other variables.
    K = means.shape[0]
    if K == 0:  # sanity check
        print("No peaks found! Adjust your parameters.")
        return [None, None, None]
    covars = np.zeros(shape=(K, 2, 2))
    pi = np.zeros(shape=K, dtype=np.float)

    # Now we need to estimate variances. Uhm... any ideas?
    for index, (i, j) in enumerate(means):
        pi[index] = image[i, j]  # We'll normalize this later.

        # We're going to do something really naive: we'll use the variance
        # of the surrounding 8 pixels as the starting value.
        i_start = i - 1 if i > 0 else 0
        i_end = i + 2 if i + 1 < image.shape[0] else image.shape[0]
        j_start = j - 1 if j > 0 else 0
        j_end = j + 2 if j + 1 < image.shape[1] else image.shape[1]

        # Assign a symmetric covariance.
        covars[index, 0, 0] = covars[index, 1, 1] = image[i_start:i_end,
                                                    j_start:j_end].flatten().var()

    pi /= pi.sum()  # Make it sum to 1.

    # All done.
    return pi, means, covars
