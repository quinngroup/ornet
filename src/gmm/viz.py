import itertools

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from scipy import linalg
import scipy.stats as stats

def plot_results(means, covariances,xmin,xmax,ymin,ymax, index, title, aff_Table = None):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange','red','yellow','green','blue','lightblue','lime'])
    splot = plt.subplot(1, 1, 1 + index)
    plt.axis([xmin, xmax, ymin, ymax])
    splot.set_aspect('equal')
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    if aff_Table is not None:
        print('not none')
        np.fill_diagonal(aff_Table,0)
        max_ind = np.argmax(aff_Table,axis=0)
        print(max_ind)
        print(max_ind.shape[0])
        for i in range(max_ind.shape[0]):
            plt.plot([means[i,0],means[max_ind[i],0]],[means[i,1],means[max_ind[i],1]],'k-')
    plt.show()

def plot_data(X, Ktrue, Kest = None):
    """
    Helper function to plot 1D data.
    """
    p = ""
    for i, (mu, sigma) in enumerate(Ktrue):
        p += "($\mu_{}$ = {:.2f}, $\sigma_{}$ = {:.2f}) ".format(
            i + 1, mu, i + 1, sigma)

    t = "K={}\n {}".format(Ktrue.shape[0], p.strip())
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("# $x_i$")
    ax1.hist(X, bins = 50)

    ax2 = ax1.twinx()
    ax2.set_ylabel("$P(x_i)$")
    x_axis = np.arange(X.min(), X.max(), 0.01)
    for i in range(Ktrue.shape[0]):
        ax2.plot(x_axis, stats.norm.pdf(x_axis, Ktrue[i][0], Ktrue[i][1]), c = 'b')

    if Kest is not None:
        for i in range(Kest.shape[0]):
            ax2.plot(x_axis, stats.norm.pdf(x_axis, Kest[i][0], Kest[i][1]), c = 'r')
    ax1.set_title(t)
    fig.tight_layout()
    plt.show()

def plot_animate_1d(X, means, vars, as_fig = None):
    """
    Creates an animation to show the convergence of the GMM.

    Parameters
    ----------
    X : array, shape (N,)
        The 1D data.
    means : array, shape (T, K)
        The means at each of T iterations.
    vars : array, shape (T, K)
        The variances at each of T iterations.
    as_fig : string
        Filepath to serialize the animation as a gif (default: None, displays figure).
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    T, K = means.shape
    fig, ax = plt.subplots()
    lines = []

    # Plot the initial values.
    x = np.arange(X.min(), X.max(), 0.01)
    ax.scatter(X, np.zeros(X.shape[0]), s = 30, alpha = 0.5, facecolors = 'none', edgecolors = 'k')
    for k in range(K):
        color = colors[k] if k < len(colors) else np.random.randint(len(colors))
        line, = ax.plot(x, stats.norm.pdf(x, means[0, k], vars[0, k]), c = color,
            label = "$\mu_{} = {:.2f}, \sigma_{} = {:.2f}$".format(k + 1, means[0, k], k + 1, vars[0, k]))
        lines.append(line)
    leg = ax.legend(loc = 0)
    ax.set_title("Iteration 0")

    def animate(t):
        for k in range(K):
            line = lines[k]
            line.set_data(x, stats.norm.pdf(x, means[t, k], vars[t, k]))
            line.set_label("$\mu_{} = {:.2f}, \sigma_{} = {:.2f}$".format(k + 1, means[t, k], k + 1, vars[t, k]))
        ax.set_title("Iteration {}".format(t))
        leg = ax.legend(loc = 0)
        return lines + [leg]

    ani = animation.FuncAnimation(fig, animate, frames = T, interval = 1000)
    if as_fig is not None:
        ani.save(as_fig)
    else:
        plt.show()
