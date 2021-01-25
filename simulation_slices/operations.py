import numpy as np

import simulation_slices.utilities as util


def get_coords_slices(coords, box_size, slice_size, slice_axis):
    """For the list of coords in box of box_size, recover the slice_idx
    for the given slice_size and slice_axis.

    Parameters
    ----------
    coords : (3, N) array
        coordinates
    box_size : float
        size of the box
    slice_size : float
        size of the slices
    slice_axis : int
        axis along which box has been sliced [x=0, y=1, z=2]

    Returns
    -------
    slice_idx : (N,) array
        index of the slice for each coordinate

    """
    slice_idx = np.floor(coords[slice_axis] / slice_size).astype(int)
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
    num_slices = box_size // slice_size

    slice_idx = get_coords_slices(
        coords=properties['coords'], box_size=box_size,
        slice_size=slice_size, slice_axis=slice_axis
    )

    # place holder to organize slice data for each property
    slice_dict = dict([(prop, [[] for _ in range(num_slices)]) for prop in properties])

    for idx in np.unique(slice_idx):
        for prop, value in properties.items():
            slice_dict[prop][idx].append(value[..., slice_idx == idx])

    return slice_dict
