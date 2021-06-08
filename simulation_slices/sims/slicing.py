from pathlib import Path
from typing import List, Union, Optional

import astropy.units as u
import h5py
import numpy as np

import simulation_slices.io as io
import simulation_slices.sims.slice_layout as slice_layout
import simulation_slices.utilities as util


def slice_file_name(
        save_dir: str, slice_axis: int, num_slices: int, snapshot: int, downsample: bool, downsample_factor: float,
) -> str:
    """Return the formatted base filename for the given number of slices."""
    fname = f"axis_{slice_axis}_nslices_{num_slices}"
    if downsample:
        fname = f"{fname}_downsample_{str(downsample_factor).replace('.', 'p')}"
    fname = f"{save_dir}/{fname}_{snapshot:03d}.hdf5"

    return fname


def create_slice_file(
    save_dir: str,
    snapshot: int,
    box_size: u.Quantity,
    z: float,
    a: float,
    h: float,
    ptypes: List[str],
    num_slices: int,
    slice_axis: int,
    maxshape: int,
    downsample: bool = False,
    downsample_factor: float = None,
) -> str:
    """Create the hdf5 file in save_dir for given slice."""
    fname = slice_file_name(
        save_dir=save_dir,
        slice_axis=slice_axis,
        num_slices=num_slices,
        snapshot=snapshot,
        downsample=downsample,
        downsample_factor=downsample_factor,
    )

    hdf_layout = slice_layout.get_slice_layout(
        num_slices=num_slices,
        slice_axis=slice_axis,
        maxshape=maxshape,
        box_size=box_size,
        snapshot=snapshot,
        ptypes=ptypes,
        z=z,
        a=a,
        h=h,
    )

    # ensure start with clean slate
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    filename = Path(fname)
    try:
        filename.unlink()
    except FileNotFoundError:
        pass
    # create hdf5 file and generate the required datasets
    io.create_hdf5(fname=filename, layout=hdf_layout, close=True)

    return str(filename)


def open_slice_file(
    save_dir: str,
    snapshot: int,
    slice_axis: int,
    num_slices: int,
    downsample: bool = False,
    downsample_factor: float = None,
    mode: str = "r",
):
    """Return the slice file with given specifications."""
    fname = slice_file_name(
        save_dir=save_dir,
        snapshot=snapshot,
        slice_axis=slice_axis,
        num_slices=num_slices,
        downsample=downsample,
        downsample_factor=downsample_factor,
    )
    h5file = h5py.File(fname, mode=mode, swmr=True)
    return h5file


def read_slice_file_properties(
    slice_file: Union[str, h5py.File],
    slice_nums: List[int],
    properties: dict,
) -> dict:
    """Read the given properties into a dict for slice_file.

    Parameters
    ----------
    slice_file : str, optional
        file to read from, determined from input if None
    slice_nums : list of int
        slice numbers to read
    properties : dict
        dsets to be loaded for each ptype [ptype: 'gas', 'dm', 'stars', 'bh']
            - ptype :
                - dsets : [dsets]
                - attrs : [attrs] => assumed at root of slice_file!

    Returns
    -------
    properties : dict with loaded dsets and attrs for each ptype and
    slice_num in properties

    """
    results = {}

    if type(slice_file) is str:
        slice_file = h5py.File(slice_file, mode="r", swmr=True)

    for ptype in properties.keys():
        results[ptype] = {}

        dsets = properties[ptype].get("dsets", None)
        attrs = properties[ptype].get("attrs", None)

        if dsets is not None:
            for dset in dsets:
                res_dset = []
                for slice_idx in slice_nums:
                    key = f"{slice_idx}/{ptype}/{dset}"
                    try:
                        h5_dset = slice_file[key]
                    except KeyError:
                        raise KeyError(f"dset {key} not found in {slice_file.filename}")

                    # for single-valued datasets we only need to load property
                    # from first non-empty slice
                    if h5_dset.attrs["single_value"]:
                        if 0 in slice_file[key].shape:
                            continue

                        res_dset.append(
                            slice_file[key][:] * u.Unit(slice_file[key].attrs["units"])
                        )
                        break

                    else:
                        # empty slice
                        if 0 in slice_file[key].shape:
                            continue

                        res_dset.append(
                            slice_file[key][:] * u.Unit(slice_file[key].attrs["units"])
                        )

                # we have a non-empty dataset
                if res_dset:
                    results[ptype][dset] = np.concatenate(res_dset, axis=-1)
                else:
                    results[ptype][dset] = None

        if attrs is not None:
            for attr in attrs:
                try:
                    results[ptype][attr] = np.atleast_1d(slice_file.attrs[attr])
                except KeyError:
                    breakpoint()
                    raise KeyError(f"attribute {attr} not found in {slice_file.filename}")

    return results


def get_coords_slices(
    coords: u.Quantity, slice_size: u.Quantity, slice_axis: int
) -> np.ndarray:
    """For the list of periodic coords recover the slice_idx for the given
    slice_size and slice_axis.

    Parameters
    ----------
    coords : (ndim, N) astropy.units.Quantity
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
    slice_idx = (coords[slice_axis] // slice_size).astype(int)
    return slice_idx


def slice_particle_list(
    box_size: u.Quantity, num_slices: int, slice_axis: int, properties: dict
) -> dict:
    """Slice the given list of (x, y, z) coordinates in slices of
    specified size along axis. Save the properties particle
    information as well.

    Parameters
    ----------
    box_size : astropy.units.Quantity
        box size
    num_slices : int
        total number of slices
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
    slice_size = box_size / num_slices

    # for each coordinate along the slice axis, determine the slice it
    # belongs to
    slice_idx = get_coords_slices(
        coords=properties["coordinates"], slice_size=slice_size, slice_axis=slice_axis
    ) % num_slices

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
