import numpy as np
import scipy.linalg as sla
import scipy.stats as stats


def normpdf(X, mu, sigma, method='direct'):
    """
    Evaluates the PDF under the current GMM parameters.

    Parameters
    ----------
    X : array, shape (N, d)
        The data.
    mu : array, shape (d,)
        Mean of the Gaussian.
    sigma : array, shape (d, d)
        Gaussian covariance.
    method : string
        'direct' is a direct implementation of the PDF. 'scipy' will use
        the scipy.stats.norm function to evaluate the PDF.

    Returns
    -------
    px : array, shape (N,)
        The probability density of each data point, given the parameters.
    """
    d = 1 if len(X.shape) == 1 else X.shape[1]
    if method == 'direct':
        # Direct implementation of the normal PDF.
        # Cross your fingers and hope there aren't any bugs.
        if d == 1:
            n = 1 / ((2 * np.pi * sigma) ** 0.5)
            e = np.exp(-(((X - mu) ** 2) / (2 * sigma)))
            px = n * e
        else:
            det = sla.det(sigma)
            inv = sla.inv(sigma)
            p = np.einsum('ni,ji,ni->n', X - mu, inv, X - mu)
            n = 1 / ((((2 * np.pi) ** d) * det) ** 0.5)
            px = np.exp(-0.5 * p) * n
    else:  # SciPy
        if d == 1:
            rv = stats.norm(mu, sigma)
        else:
            rv = stats.multivariate_normal(mu, sigma)
        px = rv.pdf(X)
    return px


def kl(X, px, m, K):
    """
    Helper function for computing the loss, aka KL-divergence.

    Parameters
    ----------
    X : array, shape (N, )
        The data.
    px : array, shape (N, )
        Ground-truth event-level probabilities for each data in X.
    m : array, shape (k, )
        Mixing coefficients for each gaussian component. Sums to 1.
    K : array, shape (k, 2)
        Means and Variances for k 1D gaussians, learned so far.

    Returns
    -------
    kl : float
        KL-divergence between the ground-truth and learned distributions.
    """
    qx = np.zeros(shape=px.shape)
    for mj, (mu, sigma) in zip(m, K):
        qx += mj * stats.norm.pdf(X, loc=mu, scale=sigma)

    # All done.
    return stats.entropy(px, qx)


def log_likelihood(X, m, mu, sigma, method='direct'):
    """
    Computes the log-likelihood of the data, given the model parameters.

    (how very frequentist of us)

    Parameters
    ----------
    X : array, shape (N, d)
        The data.
    m : array, shape (K,)
        Array of weights on each gaussian component. Sums to 1.
    mu : array, shape (K, d)
        List of Gaussian means.
    sigma : array, shape (K, d, d)
        List of Gaussian covariances.
    method : string
        Method of evaluating the normal PDF.

    Returns
    -------
    ll : float
        Log-likelihood. Lower is better; this indicates a better "fit" to
        the data with respect to the specific parameter values. Put another
        way, this indicates that the parameters do a good job of "explaining"
        the data.
    """
    N = X.shape[0]
    K = m.shape[0]

    n = np.zeros(N)
    for k in range(K):
        n += m[k] * normpdf(X, mu[k], sigma[k], method=method)
    return np.log(n).sum()
