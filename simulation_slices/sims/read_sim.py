from typing import List, Optional, Tuple

import astropy.units as u

import simulation_slices.sims.bahamas as bahamas
import simulation_slices.sims.mira_titan as mira_titan
import simulation_slices.utilities as util


SIM_SUITE_OPTIONS = ["bahamas", "miratitan"]

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


def read_particle_properties(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
    properties: List[str],
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
        "properties": properties,
        "file_num": file_num,
        "verbose": verbose,
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
