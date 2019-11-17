import numpy as np


def random_params(k=3, x_min=-5, x_max=5):
    """
    Generates random initial means and variances for each gaussian component.

    Parameters
    ----------
    k : integer
        Number of gaussian components.
    x_min : integer
        Minimum possible mean value.
    x_max : integer
        Maximum possible mean value.

    Returns
    -------
    K : array, shape (k, 2)
        List of means and variances (means in column 0, variances in column 1).
    """
    x_range = x_max - x_min
    means = x_range * np.random.random(k) + x_min
    vars = np.abs((x_range / 10) * np.random.random(k))
    return np.array(list(zip(means, vars)))


def random_data(K, nx=100):
    """
    Helper function to generate a certain amount of 1D gaussian data.

    Parameters
    ----------
    K : array, shape (k, 2)
        List of intial 1D means and variances for each gaussian.
    nx : integer
        Number of data points to generate (default: 100).

    Returns
    -------
    X : array, shape (nx,)
        List of randomly generated data, according to the given parameters.
        Points are sampled as equally as possible from each of the components.
    """
    X = np.zeros(nx)
    n_components = K.shape[0]
    for i, (mu, sigma) in enumerate(K):
        increment = nx // n_components
        start_index = i * increment
        if i == K.shape[0] - 1:
            end_index = nx
        else:
            end_index = start_index + increment
        n_draws = end_index - start_index
        X[start_index:end_index] = np.random.normal(
            loc=mu, scale=sigma, size=n_draws)
    return X
