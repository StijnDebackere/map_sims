import logging
import time
from typing import List, Optional, Tuple
from pathlib import Path

import astropy.units as u
import h5py
import mira_titan
import numpy as np
from tqdm import tqdm

import map_sims.io as io
import map_sims.utilities as util


STEP_TO_Z = {
    97: 4.000,
    121: 3.046,
    141: 2.478,
    163: 2.018,
    176: 1.799,
    189: 1.610,
    208: 1.376,
    224: 1.209,
    247: 1.006,
    279: 0.779,
    286: 0.736,
    293: 0.695,
    300: 0.656,
    307: 0.618,
    315: 0.578,
    323: 0.539,
    331: 0.502,
    338: 0.471,
    347: 0.434,
    355: 0.402,
    365: 0.364,
    382: 0.304,
    401: 0.242,
    411: 0.212,
    432: 0.154,
    453: 0.101,
    499: 0.000,
}

def get_file_nums(
    sim_dir: str,
    snapshot: int,
) -> List[int]:
    sim_info = mira_titan.MiraTitan(
        sim_dir=sim_dir,
        snapnum=snapshot,
        verbose=False,
    )
    return sim_info.datatype_info["snap"]["nums"]


def read_particle_properties(
    sim_dir: str,
    snapshot: int,
    properties: List[str] = None,
    ptype: str = None,
    file_num: int = None,
    verbose: bool = False,
    logger: util.LoggerType = None,
):
    prop_options = ["coordinates", "masses"]

    if properties is None:
        return {}

    valid_props = set(prop_options) & set(properties)
    if not valid_props:
        raise ValueError(f"properties should be in {prop_options=}")

    sim_info = mira_titan.MiraTitan(
        sim_dir=sim_dir,
        snapnum=snapshot,
        verbose=verbose,
    )
    h = sim_info.cosmo["h"]

    props = {}
    if "coordinates" in valid_props:
        if logger:
            logger.info(f"Reading coordinates for {sim_dir=} and {snapshot=}")
        data = sim_info.read_properties(
            datatype="snap",
            props=["x", "y", "z"],
            num=file_num,
        )
        if logger:
            logger.info(f"Finished reading coordinates for {sim_dir=} and {snapshot=}")

        props["coordinates"] = np.vstack(
            [data["x"], data["y"], data["z"]]
        ).T.to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))

    if "masses" in valid_props:
        props["masses"] = np.atleast_1d(sim_info.simulation_info["snap"]["m_p"]).to(
            "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
        )

    return props


def read_simulation_attributes(
    sim_dir: str,
    snapshot: int,
    attributes: List[str] = None,
    ptype: str = None,
    file_num: int = None,
    verbose: bool = False,
) -> dict:
    attr_options = ["z", "h"]

    if attributes is None:
        return {}

    valid_attrs = set(attributes) & set(attr_options)
    if not valid_attrs:
        ValueError(f"{attributes.keys()=} should be in {attr_options=}")

    sim_info = mira_titan.MiraTitan(
        sim_dir=sim_dir,
        snapnum=snapshot,
        verbose=verbose,
    )

    attrs = {}
    if "z" in valid_attrs:
        attrs["z"] = sim.z
    if "h" in valid_attrs:
        attrs["h"] = sim.cosmo["h"]

    return attrs


def read_simulation_cosmo(
    sim_dir: str,
    snapshot: int,
    cosmo: List[str] = None,
    ptype: str = None,
    file_num: int = None,
    verbose: bool = False,
) -> dict:
    cosmo_options = [
        "omega_m",
        "omega_b",
        "omega_nu",
        "sigma_8",
        "A_s",
        "h",
        "n_s",
        "w0",
        "wa",
    ]
    if cosmo is None:
        return {}

    valid_cosmo = set(cosmo) & set(cosmo_options)
    if not valid_cosmo:
        ValueError(f"{cosmo.keys()=} should be in {cosmo_options=}")

    sim = sim_dir.split("/")[-1]
    cosmo_prms = mira_titan.cosmo.cosmo_dict(
        cosmo=mira_titan.cosmo.GRID_COSMO[sim]
    )

    prms = {}
    for prm in valid_cosmo:
        prms[prm] = cosmo_prms[prm]

    return prms


def save_halo_info_file(
    sim_dir: str,
    snapshot: int,
    mass_range: Tuple[u.Quantity, u.Quantity],
    coord_range: u.Quantity = None,
    save_dir: Optional[str] = None,
    info_fname: Optional[str] = "",
    sample_haloes_bins: Optional[dict] = None,
    halo_sample: Optional[str] = None,
    logger: util.LoggerType = None,
    **kwargs,
) -> str:
    """For snapshot of simulation in sim_dir, save coordinates of haloes
    within group_range.

    Parameters
    ----------
    sim_dir : str
        path of the simulation
    snapshot : int
        snapshot to look at
    mass_range : (min, max) tuple
        minimum and maximum value for masses
    coord_range : (3, 2) array
        range for coordinates to include
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/maps/
    info_fname : str
        name for the coordinates file without extension

    Returns
    -------
    fname : str
        filename of coords file
    saves a set of coordinates to save_dir

    """
    sim_info = mira_titan.MiraTitan(
        sim_dir=sim_dir,
        snapnum=snapshot,
        verbose=False,
    )
    h = sim_info.cosmo["h"]
    z = sim_info.z

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fname = (save_dir / f"{info_fname}_{snapshot:03d}").with_suffix(".hdf5")

    group_data = sim_info.read_properties(
        "sod",
        [
            "sod_halo_mass",
            "sod_halo_radius",
            "sod_halo_min_pot_x",
            "sod_halo_min_pot_y",
            "sod_halo_min_pot_z",
            "sod_halo_mean_x",
            "sod_halo_mean_y",
            "sod_halo_mean_z",
            "fof_halo_tag",
        ],
    )

    # get all coordinates
    coordinates = np.vstack(
        [
            group_data["sod_halo_min_pot_x"],
            group_data["sod_halo_min_pot_y"],
            group_data["sod_halo_min_pot_z"],
        ]
    ).T.to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    com = np.vstack(
        [
            group_data["sod_halo_mean_x"],
            group_data["sod_halo_mean_y"],
            group_data["sod_halo_mean_z"],
        ]
    ).T.to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))

    group_ids = group_data["fof_halo_tag"]
    masses = group_data["sod_halo_mass"]
    radii = group_data["sod_halo_radius"].to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))

    mass_range = mass_range.to(
        masses.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    mass_selection = (masses > np.min(mass_range)) & (masses < np.max(mass_range))

    # also select coordinate range
    if coord_range is not None:
        coord_range = coord_range.to(
            coordinates.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
        )
        coord_selection = np.all(
            [
                (coordinates[:, i] > np.min(coord_range[i]))
                & (coordinates[:, i] < np.max(coord_range[i]))
                for i in range(coord_range.shape[0])
            ],
            axis=0,
        )
        selection = mass_selection & coord_selection
    else:
        selection = mass_selection

    # subsample the halo sample
    if sample_haloes_bins is not None:
        mass_bin_edges = sample_haloes_bins["mass_bin_edges"]
        n_in_bin = sample_haloes_bins["n_in_bin"]

        # group halo indices by mass bins
        sampled_ids = util.groupby(
            np.arange(0, masses.shape[0]), masses, mass_bin_edges
        )

        selection = []
        for i, ids in sampled_ids.items():
            # get number of haloes to draw for bin
            n = n_in_bin[i]
            if n >= len(ids):
                selection.append(ids)
            else:
                selection.append(np.random.choice(ids, size=n, replace=False))
        selection = np.concatenate(selection)

    # make relaxation cut
    halo_sample_options = ["relaxed", "unrelaxed"]
    if halo_sample is not None:
        if halo_sample == "relaxed":
            # Neto+2007 relaxation criterion
            relaxed = (
                np.sqrt(np.sum((com - coordinates) ** 2, axis=-1)) / radii
            ) < 0.07
            selection = selection & relaxed
        elif halo_sample == "unrelaxed":
            unrelaxed = (
                np.sqrt(np.sum((com - coordinates) ** 2, axis=-1)) / radii
            ) >= 0.07
            selection = selection & unrelaxed
        else:
            raise ValueError(f"{halo_sample=} not in {halo_sample_options=}")

    # only included selection
    coordinates = coordinates[selection]
    radii = radii[selection]
    masses = masses[selection].to(
        "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    group_ids = group_ids[selection].astype(int)

    data = {
        "attrs": {
            "description": "File with selected coordinates for maps.",
        },
        "coordinates": coordinates,
        "mass_range": mass_range,
        "group_ids": group_ids,
        "masses": masses,
        "radii": radii,
        "z": z,
    }

    if coord_range is not None:
        data["coord_range"] = coord_range
    if halo_sample is not None:
        data["halo_sample"] = halo_sample
    if sample_haloes_bins is not None:
        data["sample_haloes_bins"] = sample_haloes_bins

    io.dict_to_hdf5(fname=fname, data=data, overwrite=True)
    return str(fname)
