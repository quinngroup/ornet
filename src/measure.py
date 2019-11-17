import numpy as np
from scipy.linalg import inv, det

def multivariate_kl(m1, s1, m2, s2):
    """
    Multivariate KL-divergence measure for high-dimensional gaussians. For
    use with the parameters, rather than evaluating the empirical distributions.
    See derivation:
    https://stats.stackexchange.com/questions/234757/how-to-use-kullback-leibler-divergence-if-mean-and-standard-deviation-of-of-two

    For the sake of notation: N(m1, s1) ~ p, and N(m2, s2) ~ q

    Parameters
    ----------
    m1, m2 : array, shape (d,)
        Means of the two distributions.
    s1, s2 : array, shape (d, d)
        Covariance matrices of the two distributions.

    Returns
    -------
    kl : float
        KL-divergence.
    """
    s2inv = inv(s2)
    deltamu = m2 - m1
    a = np.log(det(s2) / det(s1))
    b = np.trace(s2inv @ s1)
    c = deltamu.dot(s2inv).dot(deltamu)
    d = - m1.shape[0]
    return 0.5 * (a + b + c + d)

def multivariate_js(m1, s1, m2, s2):
    """
    Jensen-Shannon divergence measure for high-dimensional gaussians. For
    use with analytic parameters, rather than evaluating empirical distributions.
    See formulation: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    For the sake of notation: N(m1, s1) ~ p, and N(m2, s2) ~ q

    Parameters
    ----------
    m1, m2 : array, shape (d,)
        Means of the two distributions.
    s1, s2 : array, shape (d, d)
        Covariance matrices of the two distributions.

    Returns
    -------
    js : float
        JS-divergence.
    """
    pq = multivariate_kl(m1, s1, m2, s2)
    qp = multivariate_kl(m2, s2, m1, s1)
    return 0.5 * (pq + qp)
