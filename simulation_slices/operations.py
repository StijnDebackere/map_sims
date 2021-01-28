import astropy.constants as c
import astropy.units as u
import numpy as np

import simulation_slices.map_tools as map_tools
import simulation_slices.utilities as util

import pdb

def masses(masses):
    """Return the sum of the list of masses."""
    return masses


def y_sz(electron_numbers, temperatures):
    """Return the y_SZ for the list of particles."""
    norm = c.sigma_T * c.k_B / (c.c**2 * c.m_e)
    norm = norm.to(u.Mpc**2 / u.K).value
    # temperatures are given in K, will divide by pixel area in Mpc^2
    return norm * electron_numbers * temperatures


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
        coords, map_center, map_size, map_res, box_size, func,
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
    box_size : float
        periodicity of the box
    func : callable
        function that calculates observable for each particle
    props : dict of (..., N) or (1,) arrays
        properties to average, should be the kwargs of func
    num_threads : int
        number of threads to use

    Returns
    -------
    mapped : (map_extent // map_res, map_extent // map_res) array
        Sum_{i in pixel} func(**props_i) / A_pix
    """
    map_res = util.check_slice_size(slice_size=map_res, box_size=map_size)
    num_pix = int(map_size // map_res)
    A_pix = map_res**2

    # convert the coordinates to the pixel coordinate system
    map_origin = map_tools.diff(np.atleast_1d(map_center), map_size / 2, box_size)

    # compute the offsets w.r.t the map_origin, taking into account
    # periodic boundary conditions
    coords_origin = map_tools.diff(coords, map_origin.reshape(2, 1), box_size)

    # get the x and y values of the pixelated maps
    x_pix = get_coords_slices(coords=coords_origin, slice_size=map_res, slice_axis=0)
    y_pix = get_coords_slices(coords=coords_origin, slice_size=map_res, slice_axis=1)

    # map (i, j) pixels to 1D pixel id = i + j * num_pix
    pix_ids = map_tools.pixel_to_pix_id([x_pix, y_pix], num_pix)

    # only use props that are within map
    within_map = pix_ids < num_pix**2
    pix_ids = pix_ids[within_map]
    sort_order = np.argsort(pix_ids)

    # get the starting index for each sorted pixel
    unique_ids, loc_ids = np.unique(pix_ids[sort_order], return_index=True)
    pix_range = np.concatenate([loc_ids, [len(pix_ids)]])

    props = dict(
        [
            (k, v[..., within_map])
            if v.shape[-1] == len(within_map)
            # if v is a single value, apply it for all coords in pixel
            else (k, v * np.ones(within_map.sum()))
            for k, v in props.items()
        ])

    # now fill up pixel_values by performing func_sum(**props) / A_pix
    func_values = func(**props)[sort_order] / A_pix

    pixel_values = np.zeros(num_pix**2, dtype=float)
    pixel_values[unique_ids] = np.array([
        np.sum(func_values[i:j]) for i, j in zip(pix_range[:-1], pix_range[1:])
    ])

    # reshape the array to the map we wanted
    mapped = np.atleast_1d(pixel_values).reshape(num_pix, num_pix)
    return mapped
