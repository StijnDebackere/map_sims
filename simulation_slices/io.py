import sys
import warnings

import astropy.units as u
import numpy as np
import h5py

import simulation_slices.utilities as util


def create_hdf5(
    fname: str,
    layout: dict,
    close: bool = False,
    overwrite: bool = False,
    swmr: bool = False,
) -> h5py.File:
    """Create an hdf5 file with layout given by dictionary

    Parameters
    ----------
    fname : str
        filename
    layout : dict
        - 'attrs' : dict
            - 'attr' : value
        - 'dsets' : dict
            - 'dset' : dict
                Either:
                - 'shape' : tuple
                - 'maxshape' : tuple
                - 'dtype' : dtype
                - 'attrs' : dict
                    - 'attr' : value
                Or:
                - 'data': array-like
                - 'attrs' : dict
                    - 'attr' : value
    close : bool
        return closed file
    overwrite : bool
        overwrite dsets if present in fname
    swmr : bool
        return file in single write multiple read mode

    Returns
    -------
    h5file : hdf5 file
        (closed) file with given layout
    """
    fname = util.check_path(fname)
    if overwrite:
        # truncate file if overwriting
        mode = "w"
    else:
        if swmr:
            raise ValueError("cannot enable swmr and not overwrite.")
        mode = "a"

    if swmr:
        libver = "v110"
    else:
        libver = "earliest"

    # create hdf5 file and generate the required datasets
    h5file = h5py.File(str(fname), mode=mode, libver=libver)

    for attr, val in layout["attrs"].items():
        # attr not yet in h5file
        if attr not in h5file.attrs.keys():
            h5file.attrs[attr] = val
        # attr already in file but does not match
        elif val != h5file.attrs[attr]:
            raise ValueError(f"{attr=} does not match")

    for dset, val in layout["dsets"].items():
        # dset already contains data
        if "data" in val.keys():
            # dset already in h5file, cannot overwrite
            if dset in h5file.keys():
                if type(val["data"]) is u.Quantity:
                    if np.allclose(val["data"].to_value(h5file[dset].attrs["units"]), h5file[dset][()]):
                        # load dset since we will compare its attributes later
                        ds = h5file[dset]
                    else:
                        raise ValueError(f"{dset=} does not match")
                else:
                    if np.allclose(val["data"], h5file[dset][()]):
                        # val can still be unit, but loaded from hdf5 file
                        if "units" in h5file[dset].attrs.keys():
                            if "units" not in val["attrs"].keys():
                                raise ValueError(f"{dset=} is not astropy.units.Quantity")
                            elif val["attrs"]["units"] != h5file[dset].attrs["units"]:
                                raise ValueError(f"{dset=} units do not match")

                        # load dset since we will compare its attributes later
                        ds = h5file[dset]
                    else:
                        raise ValueError(f"{dset=} does not match")

            # dset not yet in h5file
            else:
                if type(val["data"]) is u.Quantity:
                    ds = h5file.create_dataset(
                        dset,
                        data=val["data"].value,
                    )
                    ds.attrs["units"] = str(val["data"].unit)

                else:
                    ds = h5file.create_dataset(
                        dset,
                        data=val["data"],
                    )

        # dset only contains shape information of data to be added
        else:
            # dset already in h5file, cannot overwrite
            if dset in h5file.keys():
                if h5file[dset].shape != val["shape"]:
                    raise ValueError(f"{dset=} shape={val['shape']} does not match {h5file[dset].shape}")
                if h5file[dset].maxshape != val["maxshape"]:
                    raise ValueError(f"{dset=} maxshape={val['maxshape']} does not match {h5file[dset].maxshape}")
                if h5file[dset].dtype != val["dtype"]:
                    raise ValueError(f"{dset=} dtype={val['dtype']} does not match {h5file[dset].dtype}")
                # load dset since we will compare its attributes later
                ds = h5file[dset]

            # dset not yet in h5file
            else:
                ds = h5file.create_dataset(
                    dset,
                    shape=val["shape"],
                    dtype=val["dtype"],
                    maxshape=val["maxshape"],
                )

        if "attrs" in val.keys():
            for attr, attr_val in val["attrs"].items():
                # add attr if not present
                if attr not in ds.attrs.keys():
                    if type(attr_val) is u.Quantity:
                        ds.attrs[attr] = attr_val.value
                        ds.attrs[f"{attr}_units"] = str(attr_val.unit)
                    elif callable(attr_val):
                        ds.attrs[attr] = attr_val.__name__
                    else:
                        try:
                            ds.attrs[attr] = attr_val
                        except TypeError:
                            warnings.warn(f"{attr=} with type={type(attr_val)} cannot be saved, skipping.")
                # attr present, cannot overwrite
                else:
                    try:
                        check_equal = (attr_val == ds.attrs[attr])
                    except ValueError:
                        check_equal = np.all(attr_val, ds.attrs[attr])

                    if not check_equal:
                        breakpoint()
                        raise ValueError(f"{attr=} does not match for dset={val}")
    if close:
        h5file.close()
    else:
        if swmr:
            h5file.swmr_mode = True

    return h5file


def add_to_hdf5(h5file: h5py.File, dataset: str, vals: u.Quantity, axis: int):
    """Append vals to axis of dataset of h5file."""
    try:
        dset = h5file[dataset]
    except KeyError:
        breakpoint()
        raise KeyError(f"{dataset} not found in {h5file.filename}")

    if "units" in dset.attrs.keys():
        unit = dset.attrs["units"]

    else:
        unit = str(vals.unit)
        dset.attrs["units"] = unit

    dset.resize(dset.shape[axis] + vals.shape[axis], axis=axis)
    sl = [slice(None)] * len(dset.shape)
    sl[axis] = slice(dset.shape[axis] - vals.shape[axis], dset.shape[axis])
    sl = tuple(sl)

    dset[sl] = vals.to_value(unit)
    if h5file.swmr_mode:
        dset.flush()
