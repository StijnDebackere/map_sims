from typing import List

import astropy.units as u


def attrs(
    num_slices: int,
    slice_axis: int,
    box_size: u.Quantity,
    snapshot: int,
    z: float,
    a: float,
):
    """Return the required hdf5 attributes for all slices."""
    return {
        "num_slices": num_slices,
        "slice_axis": slice_axis,
        "slice_size": box_size.value / num_slices,
        "box_size": box_size.value,
        "size_units": str(box_size.unit),
        "snapshot": snapshot,
        "a": a,
        "z": z,
    }


def properties(maxshape):
    """Return the standard properties expected for the slice file."""
    joint = {
        "coordinates": {
            "shape": (3, 0),
            "maxshape": (3, maxshape),
            "dtype": float,
            "attrs": {
                "description": "Particle coordinates",
                "single_value": False,
                # "units": "Mpc / littleh",
            },
        },
        "masses": {
            "shape": (0,),
            "maxshape": (maxshape,),
            "dtype": float,
            "attrs": {
                "description": "Particle masses",
                "single_value": False,
                # "units": "Msun / littleh",
            },
        },
    }
    properties = {
        "gas": {
            "temperatures": {
                "shape": (0,),
                "maxshape": (maxshape,),
                "dtype": float,
                "attrs": {
                    "description": "Particle temperatures",
                    "single_value": False,
                    # "units": "K",
                },
            },
            "densities": {
                "shape": (0,),
                "maxshape": (maxshape,),
                "dtype": float,
                "attrs": {
                    "description": "Particle mass density",
                    "single_value": False,
                    # "units": "littleh2 Msun / Mpc3",
                },
            },
            "electron_number_densities": {
                "shape": (0,),
                "maxshape": (maxshape,),
                "dtype": float,
                "attrs": {
                    "description": "Particle electron number density",
                    "single_value": False,
                    # "units": "littleh2 / Mpc3",
                },
            },
            "luminosities": {
                "shape": (0,),
                "maxshape": (maxshape,),
                "dtype": float,
                "attrs": {
                    "description": "Particle X-ray luminosity",
                    "single_value": False,
                    # "units": "Lsun",
                },
            },
            **joint,
        },
        "dm": {
            **joint,
            "masses": {
                "shape": (0,),
                "maxshape": (1,),
                "dtype": float,
                "attrs": {
                    "description": "Particle masses",
                    "single_value": True,
                    # "units": "Msun / littleh",
                },
            },
        },
        "stars": {**joint},
        "bh": {**joint},
    }
    return properties


def get_slice_layout(
    slice_axis: int,
    num_slices: int,
    box_size: u.Quantity,
    snapshot: int,
    z: float,
    a: float,
    ptypes: List[str],
    maxshape: int,
) -> dict:
    """Generate the standard layout for our slice hdf5 files.

    Parameters
    ----------
    slice_axis : int
        dimension along which box has been sliced
    num_slices : int
        total number of slices
    box_size : astropy.units.Quantity
        box size
    snapshot : int
        snapshot of the simulation
    z : float
        redshift
    a : float
        expansion factor
    ptypes : ['gas', 'dm', 'stars', 'bh']
        particle types to include
    maxshape : int
        maximum shape for each hdf5 dataset

    Returns
    -------
    layout : dict
        layout for hdf5 file
    """
    ptype_options = ["gas", "dm", "stars", "bh"]
    valid_ptypes = set([p.lower() for p in ptypes]) & set(ptype_options)

    layout = {"dsets": {}}
    layout["attrs"] = attrs(
        num_slices=num_slices,
        slice_axis=slice_axis,
        z=z,
        a=a,
        box_size=box_size,
        snapshot=snapshot,
    )

    props = properties(maxshape=maxshape)
    for ptype in valid_ptypes:
        for slice_idx in range(num_slices):
            for prop, val in props[ptype].items():
                layout["dsets"][f"{slice_idx}/{ptype}/{prop}"] = val

    return layout
