from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np

import map_sims.sims.bahamas as bahamas
import map_sims.sims.mira_titan as mira_titan
import map_sims.utilities as util


SIM_SUITE_OPTIONS = ["bahamas", "miratitan"]


def snap_to_z(
    sim_suite: str,
    snapshots: np.ndarray,
) -> List[float]:
    snapshots = np.atleast_1d(snapshots)
    if sim_suite.lower() not in SIM_SUITE_OPTIONS:
        raise ValueError(f"sim_suite should be in {SIM_SUITE_OPTIONS=}")

    if sim_suite.lower() == "bahamas":
        z = np.array([bahamas.SNAP_TO_Z[snap] for snap in snapshots])
    elif sim_suite.lower() == "miratitan":
        z = np.array([mira_titan.STEP_TO_Z[snap] for snap in snapshots])

    return z


def get_file_nums(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
) -> List[int]:
    if sim_suite.lower() not in SIM_SUITE_OPTIONS:
        raise ValueError(f"sim_suite should be in {SIM_SUITE_OPTIONS=}")

    kwargs = {
        "sim_dir": sim_dir,
        "snapshot": snapshot,
    }
    if sim_suite.lower() == "bahamas":
        file_nums = bahamas.get_file_nums(**kwargs)

    elif sim_suite.lower() == "miratitan":
        file_nums = mira_titan.get_file_nums(**kwargs)

    return file_nums


def save_halo_info_file(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
    mass_range: Tuple[u.Quantity, u.Quantity],
    coord_range: u.Quantity = None,
    coord_dset: str = None,
    mass_dset: str = None,
    radius_dset: str = None,
    extra_dsets: List[str] = None,
    save_dir: Optional[str] = None,
    info_fname: Optional[str] = "",
    sample_haloes_bins: Optional[dict] = None,
    halo_sample: Optional[str] = None,
    verbose: bool = False,
    logger: util.LoggerType = None,
):
    kwargs = {
        "sim_dir": sim_dir,
        "snapshot": snapshot,
        "mass_range": mass_range,
        "coord_range": coord_range,
        "save_dir": save_dir,
        "info_fname": info_fname,
        "sample_haloes_bins": sample_haloes_bins,
        "halo_sample": halo_sample,
        "verbose": verbose,
        "logger": logger,
    }
    if sim_suite.lower() == "bahamas":
        fname = bahamas.save_halo_info_file(
            coord_dset=coord_dset,
            mass_dset=mass_dset,
            radius_dset=radius_dset,
            extra_dsets=extra_dsets,
            **kwargs
        )
    elif sim_suite.lower() == "miratitan":
        fname = mira_titan.save_halo_info_file(**kwargs)

    return fname


def read_particle_properties(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
    properties: List[str],
    ptype: str = None,
    file_num: int = None,
    verbose: bool = False,
    logger: util.LoggerType = None,
) -> dict:
    if sim_suite.lower() not in SIM_SUITE_OPTIONS:
        raise ValueError(f"sim_suite should be in {SIM_SUITE_OPTIONS=}")

    kwargs = {
        "sim_dir": sim_dir,
        "snapshot": snapshot,
        "ptype": ptype,
        "properties": properties,
        "file_num": file_num,
        "verbose": verbose,
        "logger": logger,
    }
    if sim_suite.lower() == "bahamas":
        props = bahamas.read_particle_properties(**kwargs)
    elif sim_suite.lower() == "miratitan":
        props = mira_titan.read_particle_properties(**kwargs)

    return props


def read_simulation_attributes(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
    attributes: List[str],
    ptype: str = None,
    file_num: int = None,
    verbose: bool = False,
) -> dict:
    if sim_suite.lower() not in SIM_SUITE_OPTIONS:
        raise ValueError(f"sim_suite should be in {SIM_SUITE_OPTIONS=}")

    kwargs = {
        "sim_dir": sim_dir,
        "snapshot": snapshot,
        "ptype": ptype,
        "attributes": attributes,
        "file_num": file_num,
        "verbose": verbose,
    }
    if sim_suite.lower() == "bahamas":
        attrs = bahamas.read_simulation_attributes(**kwargs)
    elif sim_suite.lower() == "miratitan":
        attrs = mira_titan.read_simulation_attributes(**kwargs)

    return attrs


def read_simulation_cosmo(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
    cosmo: List[str],
    ptype: str = None,
    file_num: int = None,
    verbose: bool = False,
) -> dict:
    if sim_suite.lower() not in SIM_SUITE_OPTIONS:
        raise ValueError(f"sim_suite should be in {SIM_SUITE_OPTIONS=}")

    kwargs = {
        "sim_dir": sim_dir,
        "snapshot": snapshot,
        "ptype": ptype,
        "cosmo": cosmo,
        "file_num": file_num,
        "verbose": verbose,
    }
    if sim_suite.lower() == "bahamas":
        attrs = bahamas.read_simulation_cosmo(**kwargs)
    elif sim_suite.lower() == "miratitan":
        attrs = mira_titan.read_simulation_cosmo(**kwargs)

    return attrs
