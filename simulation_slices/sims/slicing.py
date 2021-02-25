from pathlib import Path

import h5py
import numpy as np

import simulation_slices.io as io
import simulation_slices.sims.slice_layout as slice_layout
import simulation_slices.utilities as util


def slice_file_name(
        save_dir, slice_axis, slice_size, snapshot, slice_num=None):
    """Return the formatted base filename for the given slice. If
    slice_num is None, base filename is returned."""
    fname = f'axis_{slice_axis}_size_{slice_size}_{int(snapshot):03d}'
    if slice_num is not None:
        fname = f'{save_dir}/{fname}_{int(slice_num):d}.hdf5'

    return fname


def create_slice_file(
        save_dir, snapshot, box_size, ptypes,
        slice_num, slice_axis, slice_size, maxshape
):
    """Create the hdf5 file in save_dir for given slice."""
    fname = slice_file_name(
        save_dir=save_dir, slice_axis=slice_axis, slice_size=slice_size,
        snapshot=snapshot, slice_num=slice_num
    )

    hdf_layout = slice_layout.get_slice_layout(
        slice_num=slice_num, slice_axis=slice_axis, slice_size=slice_size,
        maxshape=maxshape, box_size=box_size, snapshot=snapshot, ptypes=ptypes,
    )

    # ensure start with clean slate
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    filename = Path(fname)
    try:
        filename.unlink()
    except FileNotFoundError:
        pass
    # create hdf5 file and generate the required datasets
    io.create_hdf5(
        fname=filename, layout=hdf_layout, close=True
    )


def read_slice_file_properties(
        properties, save_dir, snapshot, slice_num, slice_axis, slice_size):
    """Read the given properties into a dict for slice_file.

    Parameters
    ----------
    properties : dict
        dsets to be loaded for each ptype [ptype: 'gas', 'dm', 'stars', 'bh']
            - ptype :
                - dsets : str
                - ...

    Returns
    -------
    properties : dict with loaded dset for each ptype in properties

    """
    fname = slice_file_name(
        save_dir=save_dir, slice_axis=slice_axis, slice_size=slice_size,
        snapshot=snapshot, slice_num=slice_num
    )

    results = {}
    with h5py.File(fname, 'r') as h5file:
        for ptype, dsets in properties.items():
            results[ptype] = {
                dset: h5file[f'{ptype}/{dset}'][:] for dset in dsets
            }

    return results


def get_coords_slices(coords, slice_size, slice_axis):
    """For the list of periodic coords recover the slice_idx for the given
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
    slice_idx = (coords[slice_axis] // slice_size).astype(int)
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
        'coordinates': a (3, N) array
        **extra_properties: (..., N) arrays

    Returns
    -------
    dictionary containing with keys
        'coordinates' : list of box_size / slice_size lists of coordinates belonging
                   to each slice
        **extra_properties : similar lists with other properties

    """
    # ensure all passed arguments match our expectations
    slice_axis = util.check_slice_axis(slice_axis)
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = int(box_size // slice_size)

    # for each coordinate along the slice axis, determine the slice it
    # belongs to
    slice_idx = get_coords_slices(
        coords=properties['coordinates'], slice_size=slice_size,
        slice_axis=slice_axis
    )

    # place holder to organize slice data for each property
    slice_dict = dict([(prop, [[] for _ in range(num_slices)]) for prop in properties])

    for idx in np.unique(slice_idx):
        for prop, value in properties.items():
            value = np.atleast_1d(value)
            if value.shape[-1] == len(slice_idx):
                slice_dict[prop][idx].append(value[..., slice_idx == idx])

            # either (M, 1) array or (1,) array => same for each particle
            elif value.shape[-1] == 1:
                if not slice_dict[prop][idx]:
                    slice_dict[prop][idx].append(value)

                    

    return slice_dict
