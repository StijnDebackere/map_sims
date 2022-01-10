import logging
from typing import List, Optional, Tuple, Union, Any

import astropy.units as u
import h5py
import numpy as np

import map_sims.io as io
import map_sims.sims.read_sim as read_sim


def get_r_ap_names(r_aps, r2s, rms, bg=False):
    """Get standardized names for aperture masses."""
    r_ap_names = []
    if bg:
        bg_str = "_bg"
    else:
        bg_str = ""

    for r_ap, r2, rm in zip(r_aps, r2s, rms):
        ann_str = (
            f"_R2_{str(r2.value).replace('.', 'p')}_"
            f"Rm_{str(rm.value).replace('.', 'p')}"
        )
        r_ap_names.append(
            f"m_ap_{str(r_ap.value).replace('.', 'p')}{ann_str}{bg_str}"
        )

    return r_ap_names


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
    selection : np.ndarray
        selection criterion for halos in info_files

    Returns
    -------
    results : dict
        keys: sims
        values: dict
            keys: [coordinates, radii, masses, group_ids, extra_dsets.keys()]
            values: dsets from info_file found under
                    [coordinates, radii, masses, group_ids, extra_dsets.values()]

    """
    if selection is None:
        selection = ()

    results = dict((sim, {}) for sim in sims)
    for sim, info_file in zip(sims, info_files):
        coordinates = io.read_from_hdf5(info_file, "coordinates")
        radii = io.read_from_hdf5(info_file, "radii")
        group_ids = io.read_from_hdf5(info_file, "group_ids")
        masses = io.read_from_hdf5(info_file, "masses")
        z = io.read_from_hdf5(info_file, "z")

        results[sim]["coordinates"] = coordinates[selection]
        results[sim]["radii"] = radii[selection]
        results[sim]["group_ids"] = group_ids[selection]
        results[sim]["masses"] = masses[selection]
        results[sim]["z"] = z

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
    try:
        box_size = io.read_from_hdf5(map_file, "box_size")
        map_size = io.read_from_hdf5(map_file, "map_size")
        map_pix = io.read_from_hdf5(map_file, "map_pix")
        pix_size = map_size / map_pix
        map_thickness = io.read_from_hdf5(map_file, "map_thickness")
        snapshot = io.read_from_hdf5(map_file, "snapshot")
        slice_axis = io.read_from_hdf5(map_file, "slice_axis")

        # new files save map_thickness as 1d array, having box_size under key 0
        map_full = np.empty((map_thickness.shape[0], map_pix, map_pix), dtype=float)
        for idx, thickness in enumerate(map_thickness):
            map_full[idx] = io.read_from_hdf5(map_file, f"dm_mass/{idx}")[None].value

            if idx == 0:
                unit = io.read_from_hdf5(map_file, f"dm_mass/{idx}")[None].unit

            if "DMONLY" not in sim and sim_suite.lower() == "bahamas":
                for mass_type in [f"gas_mass/{idx}", f"stars_mass/{idx}", f"bh_mass/{idx}"]:
                    map_full[idx] += io.read_from_hdf5(map_file, mass_type)[None].to_value(unit)

        map_full = np.squeeze(map_full) * unit

    except KeyError:
        with h5py.File(map_file, "r") as h5_map:
            length_units = u.Unit(str(h5_map.attrs["length_units"]))
            box_size = h5_map.attrs["box_size"] * length_units
            map_size = h5_map.attrs["map_size"] * length_units
            map_pix = h5_map.attrs["map_pix"]
            pix_size = map_size / map_pix * length_units
            map_thickness = h5_map.attrs["map_thickness"] * length_units
            snapshot = h5_map.attrs["snapshot"]
            slice_axis = h5_map.attrs["slice_axis"]

            map_full = h5_map["dm_mass"][()]
            if "DMONLY" not in sim and sim_suite.lower() == "bahamas":
                for mass_type in ["gas_mass", "stars_mass", "bh_mass"]:
                    map_full += h5_map[mass_type][()]

    z = read_sim.snap_to_z(sim_suite=sim_suite.lower(), snapshots=int(snapshot))
    metadata = {
        "box_size": box_size,
        "map_size": map_size,
        "pix_size": pix_size,
        "map_thickness": map_thickness,
        "snapshot": snapshot,
        "slice_axis": slice_axis,
        "z": z,
    }
    if logger:
        logger.debug(f"loaded map from {map_file}")

    return map_full, metadata
