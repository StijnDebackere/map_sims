from pathlib import Path
from typing import List

import astropy.units as u
import h5py

import simulation_slices.io as io

MAP_TYPE_DESCRIPTIONS = {
    "gas_mass": "Projected gas mass.",
    "dm_mass": "Projected dark matter mass.",
    "stars_mass": "Projected stellar mass.",
    "bh_mass": "Projected black hole mass.",
    "y_sz": "Sunyaev-Zel'dovich effect.",
    "lum_x_ray": "Projected X-ray luminosity.",
}

def attrs(
    slice_axis: int,
    box_size: u.Quantity,
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_pix: int,
    snapshot: int,
):
    """Return the required hdf5 attributes for all maps."""
    unit = box_size.unit
    return {
        "slice_axis": slice_axis,
        "box_size": box_size.value,
        "map_size": map_size.to_value(unit),
        "map_thickness": map_thickness.to_value(unit),
        "length_units": str(unit),
        "map_pix": map_pix,
        "snapshot": snapshot,
    }


def properties(map_types, map_pix, maxshape):
    """Return the standard properties expected for the slice file."""
    props = {
        map_type: {
            "shape": (0, map_pix, map_pix),
            "maxshape": (maxshape, map_pix, map_pix),
            "dtype": float,
            "attrs": {
                "description": MAP_TYPE_DESCRIPTIONS[map_type],
                "single_value": False,
            }
        } for map_type in map_types
    }
    return props


def get_map_layout(
    slice_axis: int,
    box_size: u.Quantity,
    map_types: List[str],
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_pix: int,
    snapshot: int,
    n_ngb: int,
    maxshape: int,
    extra: dict = {},
) -> dict:
    """Generate the standard layout for our slice hdf5 files.

    Parameters
    ----------
    slice_axis : int
        dimension along which box has been sliced
    box_size : astropy.units.Quantity
        box size
    map_types : ['gas_mass', 'dm_mass', 'stellar_mass', 'bh_mass', 'y_sz']
        type of map to compute
    map_size : astropy.units.Quantity
        size of the map in units of box_size
    map_thickness : astropy.units.Quantity
        thickness of the map projection
    map_pix : int
        resolution of the map
    snapshot : int
        snapshot of the simulation
    n_ngb : int
        number of neighbours for SPH smoothing length
    maxshape : int
        maximum shape for each hdf5 dataset
    extra : dict
        extra information to save at top level of hdf5 file

    Returns
    -------
    layout : dict
        layout for hdf5 file
    """
    map_type_options = MAP_TYPE_DESCRIPTIONS.keys()
    valid_map_types = set([mp.lower() for mp in map_types]) & set(map_type_options)

    layout = {"dsets": {**extra}}
    layout["attrs"] = attrs(
        slice_axis=slice_axis,
        box_size=box_size,
        map_size=map_size,
        map_thickness=map_thickness,
        map_pix=map_pix,
        snapshot=snapshot,
    )

    props = properties(
        map_types=map_types,
        map_pix=map_pix,
        maxshape=maxshape
    )

    for map_type in valid_map_types:
        layout["dsets"][map_type] = props[map_type]

    return layout


def create_map_file(
    map_name: str,
    slice_axis: int,
    box_size: u.Quantity,
    map_types: List[str],
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_pix: int,
    snapshot: int,
    n_ngb: int,
    maxshape: int,
    extra: dict = {},
    overwrite: bool = False,
    close: bool = False,
    swmr: bool = False,
) -> h5py.File:
    """Create map_file with map_name and correct layout.

    Parameters
    ----------
    slice_axis : int
        dimension along which box has been sliced
    box_size : astropy.units.Quantity
        box size
    map_types : ['gas_mass', 'dm_mass', 'stellar_mass', 'bh_mass', 'y_sz']
        type of map to compute
    map_size : astropy.units.Quantity
        size of the map in units of box_size
    map_thickness : astropy.units.Quantity
        thickness of the map projection
    map_pix : int
        resolution of the map
    snapshot : int
        snapshot of the simulation
    n_ngb : int
        number of neighbours for SPH smoothing length
    maxshape : int
        maximum shape for each hdf5 dataset
    extra : dict
        extra information to save at top level of hdf5 file
    overwrite : bool
        overwrite map_file if already exists
    close : bool
        close map_file
    swmr : bool
        enable single writer multiple reader mode for map_file

    Returns
    -------
    map_file : h5py.File
        (closed) created map_file
    """
    if Path(map_name).exists() and not overwrite:
        if swmr:
            raise ValueError("cannot enable swmr and not overwrite")
        map_file = h5py.File(map_name, "a")
    else:
        map_layout = get_map_layout(
            slice_axis=slice_axis,
            box_size=box_size,
            map_types=map_types,
            map_size=map_size,
            map_thickness=map_thickness,
            map_pix=map_pix,
            snapshot=snapshot,
            n_ngb=n_ngb,
            maxshape=maxshape,
            extra=extra,
        )
        map_file = io.create_hdf5(
            fname=map_name,
            layout=map_layout,
            close=close,
            overwrite=overwrite,
            swmr=swmr,
        )

    if close:
        map_file.close()

    return map_file
