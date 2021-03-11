from pathlib import Path

import astropy.units as u
import h5py
import numpy as np

import simulation_slices.io as io
import simulation_slices.sims.slice_layout as slice_layout
import simulation_slices.utilities as util


def slice_file_name(
        save_dir: str, slice_axis: int, slice_size: float, snapshot: int) -> str:
    """Return the formatted base filename for the given slice."""
    fname = f'axis_{slice_axis}_size_{slice_size}_{snapshot:03d}'
    fname = f'{save_dir}/{fname}.hdf5'

    return fname


def create_slice_file(
        save_dir, snapshot, box_size, ptypes,
        num_slices, slice_axis, slice_size, maxshape):
    """Create the hdf5 file in save_dir for given slice."""
    fname = slice_file_name(
        save_dir=save_dir, slice_axis=slice_axis, slice_size=slice_size,
        snapshot=snapshot
    )

    hdf_layout = slice_layout.get_slice_layout(
        num_slices=num_slices, slice_axis=slice_axis, slice_size=slice_size,
        maxshape=maxshape, box_size=box_size, snapshot=snapshot, ptypes=ptypes,
        z=z, a=a,
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


def open_slice_file(
        save_dir: str, snapshot: int, slice_axis: int, slice_size: int,
        mode: str='r'):
    """Return the slice file with given specifications."""
    fname = slice_file_name(
        save_dir=save_dir, snapshot=snapshot,
        slice_axis=slice_axis, slice_size=slice_size,
    )
    h5file = h5py.File(fname, mode=mode)
    return h5file


def read_slice_file_properties(
        slice_nums, properties, slice_file=None,
        save_dir=None, snapshot=None, slice_size=None, slice_axis=None):
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
    properties : dict with loaded dset for each ptype and slice_num in properties

    """
    results = {}
    if slice_file is None:
        fname = slice_file_name(
            save_dir=save_dir, snapshot=snapshot,
            slice_axis=slice_axis, slice_size=slice_size,
        )
        slice_file = h5py.File(fname)
    else:
        for ptype, dsets in properties.items():
            results[ptype] = {}
            for dset in dsets:
                res_dset = []
                for slice_idx in slice_nums:
                    try:
                        key = f'{slice_idx}/{ptype}/{dset}'
                        h5_dset = slice_file[key]
                    except KeyError:
                        breakpoint()
                        raise KeyError(f'key {key} not found in {slice_file.filename}')
                    if h5_dset.attrs['single_value']:
                        res_dset.append(slice_file[f'{slice_nums[0]}/{ptype}/{dset}'][:])
                        break
                    else:
                        res_dset.append(slice_file[f'{slice_idx}/{ptype}/{dset}'][:])

                results[ptype][dset] = np.concatenate(res_dset, axis=-1)

    return results


def get_coords_slices(
        coords: np.ndarray, slice_size: float, slice_axis: int) -> np.ndarray:
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
        box_size: u.Quantity, slice_size: u.Quantity, slice_axis: int,
        properties: dict) -> dict:
    """Slice the given list of (x, y, z) coordinates in slices of
    specified size along axis. Save the properties particle
    information as well.

    Parameters
    ----------
    box_size : astropy.units.Quantity
        box size
    slice_size : astropy.units.Quantity
        thickness of the slices
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
