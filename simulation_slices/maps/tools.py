import numpy as np


def pix_id_to_pixel(pix_id, map_pix):
    """Convert pix_id = i + j * map_pix  to pixel (i, j)."""
    if ((pix_id >= map_pix**2) | (pix_id < 0)).any():
        raise ValueError('pix_id should be in [0, map_pix**2)')
    return np.array([pix_id % map_pix, pix_id // map_pix])


def pixel_to_pix_id(pixel, map_pix):
    """Convert pixel (i, j) to pix_id = i + j * map_pix."""
    pixel = np.atleast_2d(pixel)
    if pixel.shape[0] != 2 and len(pixel.shape) != 2:
        raise ValueError('pixel should be (2, n) array')
    if ((pixel >= map_pix) | (pixel < 0)).any():
        raise ValueError('mapping is only 1-1 for i,j in [0, map_pix)')

    return pixel[0] + pixel[1] * map_pix


def min_diff(x, y, box_size):
    """Return the minimum vector x - y, taking into account periodic boundary
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
    return np.mod(x - y + box_size / 2, box_size) - box_size / 2


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
    dx = min_diff(x=x, y=y, box_size=box_size)
    return np.linalg.norm(dx, axis=axis)
