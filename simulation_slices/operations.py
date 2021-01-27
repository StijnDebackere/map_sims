import astropy.constants as c
import astropy.units as u
import numpy as np

import simulation_slices.utilities as util

def sum_masses(masses):
    """Return the sum of the list of masses."""
    return sum(masses)


def sum_y_sz(electron_numbers, temperatures):
    """Return the y_SZ for the list of particles."""
    norm = c.sigma_T * c.k_B / (c.c**2 * c.m_e)
    norm = norm.to(u.Mpc**2 / u.K).value
    # temperatures are given in K, will divide by pixel area in Mpc^2
    return norm * sum(electron_numbers * temperatures)


def get_coords_slices(coords, slice_size, slice_axis, origin=None):
    """For the list of coords recover the slice_idx for the given
    slice_size and slice_axis.

    Parameters
    ----------
    coords : (ndim, N) array
        coordinates
    slice_size : float
        size of the slices
    slice_axis : int
        dimension along which box has been sliced
    origin : float
        origin to compute slices with respect to

    Returns
    -------
    slice_idx : (N,) array
        index of the slice for each coordinate

    """
    if origin is None:
        origin = 0
    slice_idx = np.floor((coords[slice_axis] - origin) / slice_size).astype(int)
    return slice_idx


def slice_particle_list(
        box_size, slice_size, slice_axis, properties):
    """Slice the given list of (x, y, z) coordinates in slices of
    specified size along axis. Save the properties particle
    information as well.

    Parameters
    ----------
    box_size : float
        box size
    slice_size : float
        thickness of the slices in same units as box_size
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    properties : dict of (..., N) array-like
        'coords': a (3, N) array
        **extra_properties: (..., N) arrays

    Returns
    -------
    dictionary containing with keys
        'coords' : list of box_size / slice_size lists of coordinates belonging
                   to each slice
        **extra_properties : similar lists with other properties

    """
    # ensure all passed arguments match our expectations
    slice_axis = util.check_slice_axis(slice_axis)
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = int(box_size // slice_size)

    slice_idx = get_coords_slices(
        coords=properties['coords'], slice_size=slice_size,
        slice_axis=slice_axis, origin=0
    )

    # place holder to organize slice data for each property
    slice_dict = dict([(prop, [[] for _ in range(num_slices)]) for prop in properties])

    for idx in np.unique(slice_idx):
        for prop, value in properties.items():
            value = np.atleast_1d(value)
            if value.shape[-1] == len(slice_idx):
                slice_dict[prop][idx].append(value[..., slice_idx == idx])
            elif value.shape[-1] == 1:
                slice_dict[prop][idx].append(value)

    return slice_dict


def coords_to_map(
        coords, map_center, map_size, map_res, func_sum, num_threads=1,
        **props):
    """Convert the given 2D coordinates to a pixelated map.

    Parameters
    ----------
    coords : (2, N) array
        (x, y) coordinates
    map_center : (2,) array
        center of the (x, y) coordinate system
    map_size : float
        size of the map
    map_res : float
        resolution of a pixel
    func_sum : callable
        function that takes props as arguments and sums their values in some way
        ensure that it also handles the case of empty properties in case the pixel
        is empty
    props : dict of (..., N) or (1,) arrays
        properties to average, should be the kwargs of func_sum
    num_threads : int
        number of threads to use

    Returns
    -------
    mapped : (map_extent // map_res, map_extent // map_res) array
        func_avg(props) in each pixel
    """
    map_res = util.check_slice_size(slice_size=map_res, box_size=map_size)
    num_pix = int(map_size // map_res)
    A_pix = map_res**2

    # convert the coordinates to the pixel coordinate system
    map_origin = (np.atleast_1d(map_center) - map_size / 2)
    coords_pix = coords - map_origin.reshape(2, 1)

    # get the x and y values of the pixelated maps
    x_pix = get_coords_slices(coords=coords_pix, slice_size=map_res, slice_axis=0)
    y_pix = get_coords_slices(coords=coords_pix, slice_size=map_res, slice_axis=1)

    # map (i, j) pixels to 1D pixel id = i + j * num_pix
    pix_ids = x_pix + y_pix * num_pix

    # get lists of all coords and props belonging to each pix_id
    coords_sort = [
        coords[..., pix_ids == idx] if ((pix_ids == idx).sum() > 0)
        else np.nan
        for idx in range(num_pix**2)
    ]
    props_sort = [
        dict(
            [
                (k, v[..., pix_ids == idx])
                if v.shape[-1] == len(pix_ids)
                # if v is a single value, apply it for all coords in pixel
                else (k, np.ones((pix_ids == idx).sum()) * v)
                for k, v in props.items()
            ])
        for idx in range(num_pix**2)
    ]

    # now fill up pixel_values by performing func_sum(**props) / A_pix
    pixel_values = np.empty(num_pix**2, dtype=float)
    for idx, (c, props) in enumerate(zip(coords_sort, props_sort)):
        pixel_value = func_sum(**props) / A_pix
        if pixel_value:
            pixel_values[idx] = pixel_value
        else:
            pixel_values[idx] = np.nan

    # reshape the array to the map we wanted
    mapped = np.atleast_1d(pixel_values).reshape(num_pix, num_pix)
    return mapped
