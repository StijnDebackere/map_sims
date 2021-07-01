import astropy.units as u
import numpy as np

import simulation_slices.utilities as util


def pix_dist(a, b, b_is_pix=True):
    """Calculate the distance between pixel centers of list of pixels
    a = (i, j) and b = (k, l).

    Parameters
    ----------
    a : (n, 2) array-like
        list of pixels
    b : (2, ) array-like
        pixel offset
    b_is_pix : bool
        b should be treated as pixel, otherwise used as coordinate

    """
    a = np.atleast_2d(a).astype(int)
    b = np.atleast_1d(b)
    if b_is_pix:
        b = b.astype(int)
    else:
        b = b.astype(float)

    if len(a.shape) > 2 or a.shape[-1] != 2:
        raise ValueError("a should be broadcastable to shape (n, 2)")
    if len(b.shape) > 1 or b.shape[-1] != 2:
        raise ValueError("b should be broadcastable to shape (2,)")

    # distance between pixels
    if b_is_pix:
        dist = np.linalg.norm(a - b, axis=-1)
    # b can be partial pixel coordinate -> need to convert a to pixel centers
    else:
        dist = np.linalg.norm(a + 0.5 - b, axis=-1)

    return dist


def pix_id_to_pixel(pix_id, map_pix):
    """Convert pix_id = i * map_pix + j to pixel (i, j) for
    (map_pix, map_pix) array.

    Parameters
    ----------
    pix_id : (n,) array-like
        list of pixel ids that uniquely map to pixel value of (map_pix, map_pix)
        array
    map_pix : int
        side length of map

    Returns
    -------
    pixels : (2, n) array-like
        (i, j) pixel values corresponding to each pix_id
        row 0: i values
        row 1: j values

    """
    pix_id = np.atleast_1d(pix_id)
    if len(pix_id.shape) > 1 or pix_id.dtype != int:
        raise ValueError("pix_id should be flat array of ints")
    if ((pix_id >= map_pix**2) | (pix_id < 0)).any():
        raise ValueError('pix_id should be in [0, map_pix**2)')
    return np.array([pix_id // map_pix, pix_id % map_pix])


def pixel_to_pix_id(pixels, map_pix):
    """Convert pixel (i, j) to pix_id = i * map_pix + j for
    (map_pix, map_pix) array.

    Parameters
    ----------
    pixels : (2, n) array-like
        (i, j) pixel values corresponding to each pix_id
        row 0: i values
        row 1: j values
    map_pix : int
        side length of map

    Returns
    -------
    pix_id : (n,) array-like
        list of pixel ids = i * map_pix + j that uniquely map to pixel
        value (i, j) of (map_pix, map_pix) array

    """
    pixels = np.atleast_2d(pixels)
    if pixels.shape[0] != 2 or len(pixels.shape) != 2:
        raise ValueError('pixels should be (2, n) array')
    if ((pixels >= map_pix) | (pixels < 0)).any():
        raise ValueError('mapping is only 1-1 for i,j in [0, map_pix)')

    return pixels[0] * map_pix + pixels[1]


def pix_id_array_to_map(pix_id_array):
    """Convert map_pix**2 array to (map_pix, map_pix) map with map[i, j]
    equal to pixel (i, j) with pix_id = i * map_pix + j.

    Parameters
    ----------
    pix_id_array : (map_pix ** 2,) array-like
        array of values for pix_id = i * map_pix + j for each pixel (i, j)

    Returns
    -------
    array : (map_pix, map_pix) array-like
        map of pix_id_array values

    """
    # pix_id_array -> values for pix_ids [0, 1, 2, ..., map_pix**2 - 1]
    pix_id_array = np.atleast_1d(pix_id_array)
    if len(pix_id_array.shape) > 1:
        raise ValueError("pix_id_array should be flattened array.")

    map_pix = pix_id_array.shape[0] ** 0.5
    if map_pix % 1 != 0:
        raise ValueError("pix_id_array should have map_pix ** 2 shape")
    map_pix = int(map_pix)

    # pix_ids follow reshape order
    mp = pix_id_array.reshape(map_pix, map_pix)
    return mp


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
    return np.linalg.norm(np.atleast_1d(dx), axis=axis)


def distances_from_centers(
    center, map_size, pix_size, box_size,
):
    """Calculate the distance for a pixel grid with pix_size of physical
    size (map_size, map_size) around center for a simulation with box_size
    (needed for periodic boundary conditions).

    Parameters
    ----------
    center : (2, ) astropy.units.Quantity
        center for the grid
    map_size : astropy.units.Quantity
        physical size of the grid
    pix_size : astropy.units.Quantity
        physical size of a pixel
    box_size : astropy.units.Quantity
        size of the simulation box

    Returns
    -------
    pix_grid : (2, n) array-like
        x, y pixel coordinates for the pixel grid
    distances : (n, ) array-like
        physical distance from center for each pixel in pix_grid
    """
    map_pix = int(box_size / pix_size)
    center = np.atleast_1d(center)

    # lower (x, y) for pixel grid => these can exceed box_size since we only care about distances around center
    n_pix = np.ceil(map_size / pix_size).astype(int)
    lower = np.floor((center - 0.5 * map_size) / pix_size).astype(int)

    # pix_x_range and pix_y_range
    pix_ranges = np.linspace(lower, lower + n_pix - 1, n_pix).T
    # (n, 2) array with column 0: x and column 1: y
    pix_grid = util.arrays_to_coords(*pix_ranges).astype(int)

    distances = pix_dist(a=pix_grid, b=center / pix_size, b_is_pix=False)
    return (pix_grid % map_pix).T, distances * pix_size


def slice_around_center(
    center: u.Quantity,
    distance: u.Quantity,
    box_size: u.Quantity,
    pix_size: u.Quantity = None,
) -> dict:
    """Return distance bounds for all coordinates that are within distance
    from center for a periodic box of box_size.

    Parameters
    ----------
    center : (dim,) astropy.units.Quantity
        coordinates
    distance : (dim,) astropy.units.Quantity
        distance from center along each dimension
    box_size : astropy.units.Quantity
        size of the simulation box
    pix_size : astropy.units.Quantity
        physical size of a pixel

    Returns
    -------
    bounds : (1, 2) or (2, 2) array
        lower and upper bounds
    """
    center = np.atleast_1d(center)
    distance = np.atleast_1d(distance)

    if len(center.shape) != 1:
        raise ValueError("center should have shape (ndim,)")

    if distance.shape[0] != center.shape[0]:
        raise ValueError("distance should match ndim of center along axis 0")

    def get_bounds(c, d, box_size, pix_size):
        unit = box_size.unit
        if d >= 0.5 * box_size:
            if pix_size is not None:
                return np.array([[0, int(box_size / pix_size)]])
            else:
                return np.array([[0, box_size.to_value(unit)]]) * unit

        if pix_size is not None:
            lower_lim = 0
            upper_lim = box_size / pix_size

            lower_bound = int((c - d) / pix_size)
            upper_bound = int((c + d) / pix_size) + 1

            unit = 1

        else:
            lower_lim = 0
            upper_lim = box_size.to_value(unit)

            lower_bound = (c - d).to_value(unit)
            upper_bound = (c + d).to_value(unit)

        if lower_bound >= lower_lim and upper_bound < upper_lim:
            return np.array([[lower_bound, upper_bound]]) * unit

        return np.array(
            [
                [lower_lim, np.mod(upper_bound, upper_lim)],
                [np.mod(lower_bound, upper_lim), upper_lim]
            ]
        ) * unit

    bounds = {
        i: get_bounds(c, d, box_size, pix_size)
        for i, (c, d) in enumerate(zip(center, distance))
    }
    return bounds
