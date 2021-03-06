from typing import Tuple

import astropy.units as u
import numpy as np

import map_sims.utilities as util


def pix_dist(a, b, b_is_pix=True):
    """Calculate the distance between pixel centers of list of pixels
    a and b.

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

    if len(a.shape) > 2 or a.shape[1] != 2:
        raise ValueError("a should be broadcastable to shape (n, 2)")
    if len(b.shape) > 1 or b.shape[0] != 2:
        raise ValueError("b should be broadcastable to shape (2,)")

    # distance between pixels
    if b_is_pix:
        dist = np.linalg.norm(a - b, axis=1)
    # b can be partial pixel coordinate -> need to convert a to pixel centers
    else:
        dist = np.linalg.norm(a + 0.5 - b, axis=1)

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
    pixels : (n, 2) array-like
        (i, j) pixel values corresponding to each pix_id
        column 0: i values
        column 1: j values
    map_pix : int
        side length of map

    Returns
    -------
    pix_id : (n,) array-like
        list of pixel ids = i * map_pix + j that uniquely map to pixel
        value (i, j) of (map_pix, map_pix) array

    """
    pixels = np.atleast_2d(pixels)
    if pixels.shape[1] != 2 or len(pixels.shape) != 2:
        raise ValueError('pixels should be (n, 2) array')
    if ((pixels >= map_pix) | (pixels < 0)).any():
        raise ValueError('mapping is only 1-1 for i,j in [0, map_pix)')

    return pixels[:, 0] * map_pix + pixels[:, 1]


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


def shift_image(image: u.Quantity, shift: u.Quantity, pix_size: u.Quantity) -> u.Quantity:
    """Shift image center by vector shift along each axis.

    Parameters
    ----------
    image : (..., n_pix, n_pix) array-like
        image to shift
    shift : (2,) array
        shift in units of pix_size along (-2, -1) axes of image
    pix_size : astropy.units.Quantity
        pixel size

    Returns
    -------
    image_shifted : (..., n_pix, n_pix) array-like
        image shifted by shift
    """
    dims = image.shape[-2:]

    indices = np.meshgrid(
        *[
            np.fft.fftfreq(n, d=pix_size) for n in dims
        ],
        indexing="ij",
    )

    # align dimension axis with shift
    k_i = np.swapaxes(2 * np.pi * np.array(indices), axis1=-2, axis2=-1) / pix_size.unit
    image_fft_shift = np.fft.fft2(
        np.fft.ifftshift(image, axes=(-2, -1)),
        axes=(-2, -1),
    ) * np.exp(1j * np.sum(k_i * shift, axis=-1))


    image_shifted = np.fft.fftshift(
        np.abs(np.fft.ifft2(image_fft_shift, axes=(-2, -1))),
        axes=(-2, -1),
    )
    return image_shifted


def average_from_image(
    R: u.Quantity,
    image: u.Quantity,
    pix_size: u.Quantity,
    R_bins: u.Quantity,
    fun=np.median,
    **kwargs,
):
    """Return the tangentially average profile from image in R_bins around center."""
    image_avg = np.zeros(R_bins.shape[0] - 1, dtype=float) * image.unit
    for idx, (R_min, R_max) in enumerate(zip(R_bins[:-1], R_bins[1:])):
        selection = (R > R_min) & (R <= R_max)
        image_avg[idx] = fun(image[selection], **kwargs)

    return image_avg


def get_coords_slices(
    coords: u.Quantity, slice_size: u.Quantity, slice_axis: int
) -> np.ndarray:
    """For the list of periodic coords recover the slice_idx for the given
    slice_size and slice_axis.

    Parameters
    ----------
    coords : (n, ndim) astropy.units.Quantity
        coordinates
    slice_size : astropy.units.Quantity
        size of the slices
    slice_axis : int
        dimension along which box has been sliced

    Returns
    -------
    slice_idx : (N,) array
        index of the slice for each coordinate

    """
    if len(coords.shape) > 2 or coords.shape[1] > 3:
        raise ValueError("coords should be of shape (n, ndim)")

    slice_idx = (coords[:, slice_axis] // slice_size).astype(int)
    return slice_idx


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

    Returns
    -------
    diff : array-like
        vector x - y

    """
    return np.mod(x - y + box_size / 2, box_size) - box_size / 2


def dist(x, y, box_size, axis=-1):
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
    axis : int
        axis along which dimensions are defined

    Returns
    -------
    dist : float
        distance between x and y

    """
    dx = min_diff(x=x, y=y, box_size=box_size)
    return np.linalg.norm(np.atleast_1d(dx), axis=axis)


def distances_from_center(
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
    pix_grid : (n, 2) array-like
        x, y pixel coordinates for the pixel grid with center at n // 2
    distances : (n, ) array-like
        physical distance from center for each pixel in pix_grid
    """
    map_pix = int(box_size / pix_size)
    center = np.atleast_1d(center)

    if len(center.shape) > 1:
        raise ValueError("center should be 1D array")

    # find the pixel containing the central coordinate
    pix_center = np.floor(center / pix_size).astype(int)
    n_pix_half = np.ceil(0.5 * map_size / pix_size).astype(int)

    # row 0: pix_x_range, row 1: pix_y_range
    pix_ranges = np.linspace(
        pix_center - n_pix_half,
        pix_center + n_pix_half,
        2 * n_pix_half + 1,
    ).T
    # column 0: pix_x, column 1: pix_y
    pix_grid = util.arrays_to_coords(*pix_ranges).astype(int)

    distances = pix_dist(a=pix_grid, b=center / pix_size, b_is_pix=False)
    return (pix_grid % map_pix), distances * pix_size


def slice_map_around_center(
    center: u.Quantity,
    map_full: u.Quantity,
    map_cutout_size: u.Quantity,
    pix_size: u.Quantity,
    box_size: u.Quantity,
    shift_to_center: bool = False,
) -> Tuple[u.Quantity, u.Quantity]:
    """Slice (map_cutout_size, map_cutout_size) region from map_full with
    pix_size around center (in units of pix_size).

    Parameters
    ----------
    center: astropy.units.Quantity
        coordinate to center on
    map_full : (..., n_pix, n_pix) astropy.units.Quantity
        map to slice from
    map_cutout_size : astropy.units.Quantity
        size of cutout region
    pix_size : astropy.units.Quantity
        size of each pixel in map_full
    box_size : astropy.units.Quantity
        periodic side length
    shift_to_center : bool
        perform subpixel shift to fully center on center

    Returns
    -------
    map_cutout : (map_cutout_size // pix_size, map_cutout_size // pix_size) astropy.units.Quantity
        map_full cutout around center
    dists : (map_cutout_size // pix_size, map_cutout_size // pix_size) astropy.units.Quantity
        distance from pixel to center
    """
    if shift_to_center:
        shift = (center / pix_size % 1) * pix_size
        map_cutout_extra = map_cutout_size + np.max(shift)
    else:
        map_cutout_extra = 1 * map_cutout_size

    pix, dists = distances_from_center(
        center=center, map_size=map_cutout_extra, pix_size=pix_size, box_size=box_size
    )
    n_pix = int(pix.shape[0] ** 0.5)
    map_cutout = map_full[..., pix[:, 0], pix[:, 1]].reshape(-1, n_pix, n_pix)

    if shift_to_center:
        # shifts are always sub-pixel => distances
        map_cutout = shift_image(map_cutout, shift=shift, pix_size=pix_size)
        map_cutout = map_cutout[..., 1:-1, 1:-1]
        dists = pix_dist(
            a=tools.arrays_to_coords(*[np.arange(i) for i in map_cutout.shape[-2:]]),
            b=np.array(map_cutout.shape[-2:]) / 2,
            b_is_pix=True,
        ) * pix_size

    map_cutout = np.squeeze(map_cutout)
    return map_cutout, dists.reshape(map_cutout.shape[-2:])


def get_slice_around_center(
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
        lower and upper bounds on possibly different sides of the box
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
