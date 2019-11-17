import argparse
import os.path
import time

import numpy as np

from loss import log_likelihood, normpdf
import text, rand, image
import params

def e_step(X, m, mu, sigma, method = 'direct'):
    """
    Implements the E-step of the GMM: estimates responsibilities.

    Parameters
    ----------
    X : array, shape (N, d)
        Array of d-dimensional data.
    m : array, shape (K,)
        Mixing coefficients, sometimes referred to as pi.
    mu : array, shape (K, d)
        Current means of the K components.
    sigma : array, shape (K, d, d)
        Covariance matrices, one for each of the K components.
    method : string
        Method used to evaluate the PDF ('direct', 'scipy').

    Returns
    -------
    r : array, shape (N, K)
        The fractional mapping of each data point to a component.
    """
    d = 1 if len(X.shape) == 1 else X.shape[1]
    N = X.shape[0]
    K = m.shape[0]
    r = np.zeros(shape = (N, K))
    denom = np.zeros(shape = (N,))
    for j in range(K):
        r[:, j] = m[j] * normpdf(X, mu[j], sigma[j], method = method)
        denom += r[:, j]
    rnorm = r / denom[:, np.newaxis]
    return rnorm

def m_step(X, r):
    """
    Implements the M-step of the GMM: recompute the model parameters.

    Parameters
    ----------
    X : array, shape (N, d)
        Array of d-dimensional data.
    r : array, shape (N, K)
        The fractional mapping of each data point to a component.

    Returns
    -------
    mu : array, shape (K, d)
        Current means of the K components.
    sigma : array, shape (K, d, d)
        Covariance matrices, one for each of the K components.
    m : array, shape (K,)
        Mixing coefficients, sometimes referred to as pi.
    """
    d = 1 if len(X.shape) == 1 else X.shape[1]
    K = r.shape[1]
    Nk = r.sum(axis = 0)  # Length K

    m_next = np.zeros(shape = (K,))
    mu_next = np.zeros(shape = K if d == 1 else (K, d))
    sigma_next = np.zeros(shape = K if d == 1 else (K, d, d))

    # Re-estimate Gaussian parameters.
    for k in range(K):
        if d == 1:
            mu_next[k] = (r[:, k] * X).sum() / Nk[k]
            sigma_next[k] = (r[:, k] * ((X - mu_next[k]) ** 2)).sum() / Nk[k]
            m_next[k] = Nk[k] / Nk.sum()
        else:
            mu_next[k] = (r[:, k][:, np.newaxis] * X).sum(axis = 0) / Nk[k]
            sigma_next[k] = np.einsum("ni,nj->ij", (X - mu_next[k]) * r[:, k][:, np.newaxis], X - mu_next[k]) / Nk[k]
            m_next[k] = Nk[k] / Nk.sum()
    return mu_next, sigma_next, m_next


def gmm_iter(X, PI, MU, CV, epsilon=.01, max_iters=100, normfunc='direct', verbose=False):
    """
    Implements the iteration portion of the gmm: run e and m steps until
    it converges or times out.

    Parameters
    ----------
    X : array, shape (N, 2)
        The data.
    PI : array, shape (k,)
        List of initial mixing coefficients.
    MU : array, shape (k, 2)
        List of initial averages.
    CV : array, shape (k, 2, 2)
        List of initial covariances.
    epsilon : float, default .01
        the Epsilon of EM convergence for log likelihood
    max_iters : int, default 100
        Maximum number of EM iterations
    normfunc : ['scipy', 'direct'], default 'direct'
        Method for computing the normal PDF

    Returns
    -------
    PI : array, shape (k,)
        List of final mixing coefficients.
    MU : array, shape (k, 2)
        List of final averages.
    CV : array, shape (k, 2, 2)
        List of final covariances.
    """

    # Run the dang belly, I mean EM.
    iterations = 0
    epsilon_cur = epsilon * 2
    ll_prev = log_likelihood(X, PI, MU, CV)
    ll = [ll_prev]
    while iterations < max_iters and epsilon_cur > epsilon:
        start = time.time()

        # E-step.
        R = e_step(X, PI, MU, CV, method=normfunc)

        # M-step.
        MU_1, CV_1, PI_1 = m_step(X, R)

        # Update everything for the next round.
        ll_curr = log_likelihood(X, PI_1, MU_1, CV_1, method=normfunc)
        MU = MU_1
        CV = CV_1
        PI = PI_1

        epsilon_cur = np.abs(ll_curr - ll_prev)
        ll_prev = ll_curr
        ll.append(ll_curr)
        iterations += 1
        end = time.time()
        if verbose:
            print("Iteration {} :: Log Likelihood {:.2f} :: Time {:.2f}s."
                .format(iterations, ll_prev, end - start))
    return (PI, MU, CV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Custom Gaussian Mixture Model for the OrNet project.',
        add_help = 'How to use')

    subparser = parser.add_subparsers(
        description = 'Because why would we make this easy?',
        prog = 'python gmm.py <subcommand>', dest = "subcommand")

    # Debug subcommand: generate synthetic data on-the-fly.
    dbg = subparser.add_parser("debug", help = "Debug mode. 1D data only.")
    dbg.add_argument("--n_samples", type = int, default = 300,
        help = "Total number of data points to draw. [DEFAULT: 300]")
    dbg.add_argument("--n_gaussians", type = int, default = 3,
        help = "Number of sample Gaussians to draw from. [DEFAULT: 3]")
    dbg.add_argument("--min_x", type = float, default = -5,
        help = "Minimum of the range of possible random data. [DEFAULT: -5]")
    dbg.add_argument("--max_x", type = float, default = 5,
        help = "Maximum of the range of possible random data. [DEFAULT: 5]")

    # Text subcommand: parse CSV data off the filesystem.
    txt = subparser.add_parser("text", help = "CSV text file input. 1D data only.")
    txt.add_argument("-i", "--input", required = True,
        help = "Path to a CSV text file containing the data.")
    txt.add_argument("-p", "--params", default = None,
        help = "Path to a CSV text file containing initial parameter values. [DEFAULT: None]")

    # Image subcommand: use a single image and generate a GMM from its PDF.
    img = subparser.add_parser("image", help = "Single image input.")
    img.add_argument("-i", "--input", required = True,
        help = "Path to single image file.")
    img.add_argument("--min_dist", type = int, default = 2,
        help = "Minimum pixel distance between initial means. [DEFAULT: 2]")

    # Video subcommand: basically the image, but over and over.
    vid = subparser.add_parser("video", help = "Video input.")
    vid.add_argument("-i", "--input", required = True,
        help = "Path to folder of images (frames) or NumPy video file.")

    # NPY subcommand: basically the image, but loading a npy file
    # (mirrors skl_gmm).
    npyf = subparser.add_parser("npy", help = "Numpy input.")
    npyf.add_argument("-i", "--input", required = True,
        help = "Path to NumPy image file.")
    npyf.add_argument("--min_dist", type = int, default = 10,
        help = "Minimum pixel distance between initial means. [DEFAULT: 2]")

    # General, optional GMM arguments.
    parser.add_argument("--normfunc", choices = ['scipy', 'direct'], default = 'direct',
        help = "Method for computing the normal PDF. [DEFAULT: 'direct']")
    parser.add_argument("-k", "--n_components", type = int, default = 3,
        help = "Number of Gaussian components. [DEFAULT: 3]")
    parser.add_argument("-x", "--max_iters", type = int, default = 100,
        help = "Maximum number of EM iterations. [DEFAULT: 100]")
    parser.add_argument("-e", "--epsilon", type = float, default = 0.01,
        help = "Epsilon of EM convergence for log-likelihood. [DEFAULT: 0.01]")
    parser.add_argument("-o", "--output", default = ".",
        help = "Output directory for figures and files.")
    parser.add_argument("--r_seed", type = int, default = None,
        help = "The random seed for generating data. [DEFAULT: None]")
    parser.add_argument("-v", "--verbose", action = "store_true",
        help = "If set, this spits out verbose output. [DEFAULT: None]")

    args = vars(parser.parse_args())
    if args['subcommand'] == 'debug':
        if args['r_seed'] is not None:
            np.random.seed(args['r_seed'])

        # Generate sample 1D gaussians.
        Ktrue = rand.random_params(args['n_gaussians'], args['min_x'], args['max_x'])

        # Sample the data for each gaussian.
        X = rand.random_data(Ktrue, args['n_samples'])

        # Initialize our model parameters.
        PI, MU, CV = params.random_init(X, args['n_components'])
        if args['verbose']:
            print(Ktrue)
    elif args['subcommand'] == 'text':
        # Read in the data.
        if args['params'] is not None:
            X, PI, MU, CV = text.load_data_and_params(args['text'], args['params'])
        else:
            if args['r_seed'] is not None:
                np.random.seed(args['r_seed'])
            X = text.load_data(args['text'])
            PI, MU, CV = params.random_init(X, args['n_components'])
    elif args['subcommand'] == 'image':
        # Load a single image.
        img = image.read_image(args['input'])
        X = image.img_to_px(img)
        PI, MU, CV = params.image_init(img, k = args['n_components'],
            min_distance = args['min_dist'])
        if PI is None:
            quit("Unable to overcome previous error, exiting...")
    elif args['subcommand'] == 'npy':
        # Load a single image.
        img = np.load(args['input'])
        X = image.img_to_px(img)
        PI, MU, CV = params.image_init(img, k=None,  # None uses max filter
                                       min_distance=args['min_dist'])
        if PI is None:
            quit("Unable to overcome previous error, exiting...")
    elif args['subcommand'] == 'video':
        pass
    else:
        print("Invalid subcommand found! Don't know what to do other than quit :(")
        raise

    # Run the dang belly, I mean EM.
    iterations = 0
    epsilon = args['epsilon'] * 2
    if args['verbose']:
        print("PI {} :: MU {} :: CV {} :: Components {}".format(PI, MU, CV, CV.shape))
    ll_prev = log_likelihood(X, PI, MU, CV)
    ll = [ll_prev]
    while iterations < args['max_iters'] and epsilon > args['epsilon']:
        start = time.time()

        # E-step.
        R = e_step(X, PI, MU, CV, method = args['normfunc'])

        # M-step.
        MU_1, CV_1, PI_1 = m_step(X, R)

        # Update everything for the next round.
        ll_curr = log_likelihood(X, PI_1, MU_1, CV_1, method = args['normfunc'])
        MU = MU_1
        CV = CV_1
        PI = PI_1

        epsilon = np.abs(ll_curr - ll_prev)
        ll_prev = ll_curr
        ll.append(ll_curr)
        iterations += 1
        end = time.time()
        if args['verbose']:
            print("Iteration {} :: Log Likelihood {:.2f} :: Time {:.2f}s."
                .format(iterations, ll_prev, end - start))
            #np.save(os.path.join(args['output'], "PI_{}.npy".format(iterations)), PI)
            #np.save(os.path.join(args['output'], "MU_{}.npy".format(iterations)), MU)
            #np.save(os.path.join(args['output'], "CV_{}.npy".format(iterations)), CV)
    np.save(os.path.join(args['output'], "ll_final.npy"), np.array(ll))
    np.save(os.path.join(args['output'], "MU_final.npy"), MU)
    np.save(os.path.join(args['output'], "CV_final.npy"), CV)
