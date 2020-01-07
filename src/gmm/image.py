import imageio
import numpy as np
import skimage
import skimage.color as color


def read_image(Ipath):
    """
    Reads an image off the filesystem and converts it to a
    probability distribution.

    Parameters
    ----------
    Ipath : string
        Path to the image file.

    Returns
    -------
    img : array, shape (H, W)
        The 8-bit grayscale image.
    """
    img = imageio.imread(Ipath)
    return skimage.img_as_ubyte(skimage.color.rgb2gray(img))


def img_to_px(image):
    """
    Converts the image to a probability distribution amenable to GMM.

    Parameters
    ----------
    image : array, shape (H, W)
        8-bit grayscale image.

    Returns
    -------
    X : array, shape (N, 2)
        The data.
    """
    # We need the actual 2D coordinates of the pixels.
    # The following is fairly standard practice for generating a grid
    # of indices, often to evaluate some function on a discrete surface.
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Now we unroll the indices and stack them into 2D (i, j) coordinates.
    z = np.vstack([yy.flatten(), xx.flatten()]).T

    # Finally, we repeat each index by the number of times of its pixel value.
    # That is our X--consider each pixel an "event", and its value is the
    # number of times that event is observed.
    X = np.repeat(z, image.flatten(), axis=0)
    return X
