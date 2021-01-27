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
