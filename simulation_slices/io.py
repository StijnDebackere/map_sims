import h5py
import sys

import pdb


def create_hdf5(fname, layout, close=False):
    """Create an empty hdf5 file with layout given by dictionary

    Parameters
    ----------
    fname : str
        filename
    layout : dict
        - 'attrs' : dict
            - 'attr' : value
        - 'dsets' : dict
            - 'dset' : dict
                - 'shape' : tuple
                - 'maxshape' : tuple
                - 'dtype' : dtype
                - 'attrs' : dict
                    - 'attr' : value

    Returns
    -------
    h5file : hdf5 file if not close
        file with given layout
    """
    # create hdf5 file and generate the required datasets
    h5file = h5py.File(str(fname), mode='a')

    for attr, val in layout['attrs'].items():
        h5file.attrs[attr] = val

    for dset, val in layout['dsets'].items():
        ds = h5file.create_dataset(
            dset, shape=val['shape'], dtype=val['dtype'],
            maxshape=val['maxshape']
        )
        for attr, attr_val in val['attrs'].items():
            ds.attrs[attr] = attr_val
    if close:
        h5file.close()

    return h5file


def add_to_hdf5(h5file, dataset, vals, axis):
    """Add vals to axis of dataset of h5file."""
    dset = h5file[dataset]
    dset.resize(
        dset.shape[axis] + vals.shape[axis], axis=axis
    )
    sl = [slice(None)] * len(dset.shape)
    sl[axis] = slice(dset.shape[axis] - vals.shape[axis], dset.shape[axis])
    sl = tuple(sl)

    dset[sl] = vals
