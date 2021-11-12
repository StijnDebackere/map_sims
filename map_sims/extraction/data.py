import logging
from typing import List, Optional, Tuple, Union, Any

import astropy.units as u
import h5py
import numpy as np

import map_sims.io as io
import map_sims.sims.read_sim as read_sim


def load_from_info_files(
    sims: List[str],
    info_files: List[str],
    extra_dsets: dict,
    selection: np.ndarray = None,
) -> dict:
    """Load coordinates, group_ids, masses and extra_dsets from
    info_file for each sim.

    Parameters
    ----------
    sims : list of str
        simulation names
    info_files : list of str
        file containing coordinates, group_ids, masses and extra_dsets for each sim
    extra_dsets : dict
        keys: name for result
        values: dset in info_files for result

    Returns
    -------
    results : dict
        keys: sims
        values: dict
            keys: [coordinates, masses, group_ids, extra_dsets.keys()]
            values: dsets from info_file found under
                    [coordinates, masses, group_ids, extra_dsets.values()]

    """
    if selection is None:
        selection = ()

    results = dict((sim, {}) for sim in sims)
    for sim, info_file in zip(sims, info_files):
        coordinates = io.read_from_hdf5(info_file, "coordinates")[selection]
        group_ids = io.read_from_hdf5(info_file, "group_ids")[selection]
        masses = io.read_from_hdf5(info_file, "masses")[selection]

        results[sim]["coordinates"] = coordinates[selection]
        results[sim]["group_ids"] = group_ids[selection]
        results[sim]["masses"] = masses[selection]

        extra = {}
        for name, dset in extra_dsets.items():
            extra[name] = io.read_from_hdf5(info_file, dset)[selection]

        results[sim] = {
            **results[sim],
            **extra,
        }

    return results


def load_map_file(
    sim: str,
    map_file: str,
    sim_suite: str = "bahamas",
    logger: logging.Logger = None,
) -> Union[Tuple[np.ndarray, dict], np.ndarray]:

    """Load full map from map_file for sim, possibly return metadata.

    Parameters
    ----------
    sim : str
        simulation name
    map_file : str
        location for map file
    return_metadata : bool [Default = True]
        return map metadata

    Returns
    -------
    map_full : array-like
        mass map for sim
    metadata : optional, dict
        metadata for given map

    """
    # read metadata from hdf5 file
    with h5py.File(map_file, "r") as h5_map:
        length_units = u.Unit(str(h5_map.attrs["length_units"]))
        box_size = h5_map.attrs["box_size"] * length_units
        pix_size = h5_map.attrs["map_size"] / h5_map.attrs["map_pix"] * length_units
        map_thickness = h5_map.attrs["map_thickness"] * length_units
        snapshot = h5_map.attrs["snapshot"]

        z = read_sim.snap_to_z(sim_suite=sim_suite.lower(), snapshots=int(snapshot))
        metadata = {
            "box_size": box_size,
            "pix_size": pix_size,
            "map_thickness": map_thickness,
            "snapshot": snapshot,
            "z": z,
        }

        # new files save map_thickness as 1d array, having box_size under key 0
        if isinstance(h5_map["dm_mass"], h5py.Group):
            path_append = "/0"
        elif isinstance(h5_map["dm_mass"], h5py.Dataset):
            path_append = ""

    map_full = io.read_from_hdf5(map_file, "dm_mass" + path_append)
    if "DMONLY" not in sim and sim_suite.lower() == "bahamas":
        for mass_type in ["gas_mass", "stars_mass", "bh_mass"]:
            map_full += io.read_from_hdf5(map_file, mass_type + path_append)

    if logger:
        logger.debug(f"loaded map from {map_file}")

    return map_full, metadata
