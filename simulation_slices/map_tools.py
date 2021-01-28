import numpy as np


def pix_id_to_pixel(pix_id, num_pix):
    """Convert pix_id to pixel (i, j)."""
    return np.array([[pix_id % num_pix], [pid_id // num_pix]])


def pixel_to_pix_id(pixel, num_pix):
    """Convert pixel (i, j) to pix_id = i * num_pix + j."""
    pixel = np.atleast_2d(pixel)
    if pixel.shape[0] != 2 and len(pixel.shape) != 2:
        raise ValueError('pixel should be (2, n) array')
    return pixel[0] + pixel[1] * num_pix


def diff(x, y, box_size):
    """Return the vector x - y, taking into account periodic boundary
    conditions.

    Parameters
    ----------
    x : array-like
        coordinates
    y : array-like
        coordinates
    box_size : float
        periodicity of the box
    axis : axis along which dimensions are defined

    Returns
    -------
    diff : array-like
        vector x - y

    """
    return np.mod(x - y, box_size)


def dist(x, y, box_size, axis=0):
    """Return the distance |x-y| taking into account periodic boundary
    conditions.

    Parameters
    ----------
    x : array-like
        coordinates
    y : array-like
        coordinates
    box_size : float
        periodicity of the box
    axis : axis along which dimensions are defined

    Returns
    -------
    dist : float
        distance between x and y

    """
    return np.linalg.norm(np.mod(x - y, box_size / 2), axis=axis)
