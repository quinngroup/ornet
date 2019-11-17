import numpy as np


def load_data(Xpath):
    """
    Helper function to load a CSV text file of the data off the filesystem.
    Basically a very thin wrapper around np.loadtxt.

    Parameters
    ----------
    Xpath : string
        Path to the text file.

    Returns
    -------
    X : array, shape (N,)
        Returns the 1D data.
    """
    return np.loadtxt(Xpath)


def load_data_and_params(Xpath, Kpath):
    """
    Simultaneously loads data and initial parameter values.
    All assumed to be 1D.

    Parameters
    ----------
    Xpath : string
        Path to the file containing the data.
    Kpath : string
        Path to the file containing initial component parameters. One component
        per row, with three elements: [pi, mu, sigma].

    Returns
    -------
    X : array, shape (N,)
        The data.
    PI : array, shape (K,)
        The mixing coefficients.
    MU : array, shape (K,)
        The means.
    CV : array, shape (K,)
        The variances.
    """
    X = load_data(Xpath)
    # Parameters are the columns, components are the rows.
    params = np.loadtxt(Kpath)
    return X, params[:, 0], params[:, 1], params[:, 2]
