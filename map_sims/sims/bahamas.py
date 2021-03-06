import logging
import time
from typing import List, Optional, Tuple

import astropy.units as u
from gadget import Gadget
import h5py
import numpy as np
from tqdm import tqdm

import map_sims.io as io
import map_sims.utilities as util


# conversion between BAHAMAS particle types and expected ones
BAHAMAS_TO_PTYPES = {0: "gas", 1: "dm", 4: "stars", 5: "bh"}
PTYPES_TO_BAHAMAS = {"gas": 0, "dm": 1, "stars": 4, "bh": 5}

# conversion between expected properties and BAHAMAS hdf5 datasets
PROPS_TO_BAHAMAS = {
    ptype: {
        "coordinates": f"PartType{PTYPES_TO_BAHAMAS[ptype]}/Coordinates",
        "masses": f"PartType{PTYPES_TO_BAHAMAS[ptype]}/Mass",
    }
    for ptype in ["gas", "dm", "stars", "bh"]
}
PROPS_TO_BAHAMAS["gas"] = {
    "temperatures": f"PartType0/Temperature",
    "densities": f"PartType0/Density",
    "smoothed_hydrogen": f"PartType0/SmoothedElementAbundance/Hydrogen",
    "smoothed_helium": f"PartType0/SmoothedElementAbundance/Helium",
    "smoothed_carbon": f"PartType0/SmoothedElementAbundance/Carbon",
    "smoothed_nitrogen": f"PartType0/SmoothedElementAbundance/Nitrogen",
    "smoothed_oxygen": f"PartType0/SmoothedElementAbundance/Oxygen",
    "smoothed_neon": f"PartType0/SmoothedElementAbundance/Neon",
    "smoothed_magnesium": f"PartType0/SmoothedElementAbundance/Magnesium",
    "smoothed_silicon": f"PartType0/SmoothedElementAbundance/Silicon",
    "smoothed_iron": f"PartType0/SmoothedElementAbundance/Iron",
    **PROPS_TO_BAHAMAS["gas"],
}
# conversion between expected attributes and BAHAMAS hdf5 datasets
ATTRS_TO_BAHAMAS = {
    "z": "Header/Redshift",
    "h": "Header/HubbleParam",
}
SNAP_TO_Z = {
    22: 2.0,
    23: 1.75,
    24: 1.5,
    25: 1.25,
    26: 1.0,
    27: 0.75,
    28: 0.5,
    29: 0.375,
    30: 0.25,
    31: 0.125,
    32: 0.0,
}




def get_file_nums(
    sim_dir: str,
    snapshot: int,
) -> List[int]:
    snap_info = Gadget(
        model_dir=sim_dir,
        file_type="snap",
        snapnum=snapshot,
        units=True,
        comoving=True,
        verbose=False,
    )
    return list(range(0, snap_info.num_files))


def read_particle_properties(
    sim_dir: str,
    snapshot: int,
    ptype: str,
    properties: List[str] = None,
    file_num: int = None,
    verbose: bool = False,
    logger: util.LoggerType = None,
) -> dict:
    ptype_options = PTYPES_TO_BAHAMAS.keys()

    if properties is None:
        return {}

    if ptype not in ptype_options:
        raise ValueError(f"{ptype=} should be in {ptype_options=}")

    snap_info = Gadget(
        model_dir=sim_dir,
        file_type="snap",
        snapnum=snapshot,
        units=True,
        comoving=True,
        verbose=verbose,
    )
    props = {}
    if ptype == "dm" and "masses" in properties:
        properties.pop(properties.index("masses"))
        props["masses"] = np.atleast_1d(snap_info.masses[PTYPES_TO_BAHAMAS[ptype]])

    for prop in properties:
        if logger:
            logger.info(f"Reading {prop=} for {ptype=} from {sim_dir=} and {snapshot=}")
        if file_num is None:
            props[prop] = snap_info.read_var(var=PROPS_TO_BAHAMAS[ptype][prop])
        else:
            props[prop] = snap_info.read_single_file(
                var=PROPS_TO_BAHAMAS[ptype][prop],
                i=file_num,
            )
        if logger:
            logger.info(f"Finished reading {prop=} for {ptype=} from {sim_dir=} and {snapshot=}")

    return props


def read_simulation_attributes(
    sim_dir: str,
    snapshot: int,
    attributes: List[str] = None,
    file_num: int = None,
    ptype: str = None,
    verbose: bool = False,
) -> dict:
    attr_options = ATTRS_TO_BAHAMAS.keys()

    if attributes is None:
        return {}

    valid_attrs = set(attributes) & set(attr_options)
    if not valid_attrs:
        raise ValueError(f"{attributes=} should be in {attr_options=}")

    snap_info = Gadget(
        model_dir=sim_dir,
        file_type="snap",
        snapnum=snapshot,
        units=True,
        comoving=True,
        verbose=verbose,
    )

    if file_num is None:
        file_num = 0

    attrs = {
        attr: snap_info.read_attr(ATTRS_TO_BAHAMAS[attr], ids=file_num)
        for attr in valid_attrs
    }
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
    coord_dset: str,
    mass_dset: str,
    radius_dset: str,
    mass_range: Tuple[u.Quantity, u.Quantity],
    coord_range: u.Quantity = None,
    extra_dsets: List[str] = None,
    save_dir: Optional[str] = None,
    info_fname: Optional[str] = "",
    sample_haloes_bins: Optional[dict] = None,
    halo_sample: Optional[str] = None,
    verbose: Optional[bool] = False,
    logger: util.LoggerType = None,
    **kwargs,
) -> str:
    """For snapshot of simulation in sim_dir, save the coord_dset for
    given group_dset and group_range.

    Parameters
    ----------
    sim_dir : str
        path of the MiraTitanU directory
    snapshot : int
        snapshot to look at
    coord_dset : str
        hdf5 dataset containing the coordinates
    mass_dset : str
        hdf5 FOF dataset to get masses from
    radius_dset : str
        hdf5 FOF dataset to get radii from
    mass_range : (min, max) tuple
        minimum and maximum value for mass_dset
    coord_range : (3, 2) array
        range for coordinates to include
    extra_dsets : iterable
        extra datasets to save to the file
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/maps/
    info_fname : str
        name for the coordinates file without extension

    Returns
    -------
    fname : str
        fname of saved coordinates

    """
    if logger:
        logger.info(f"Start saving coordinates for {sim_dir=} and {snapshot=}")

    group_info = Gadget(
        model_dir=sim_dir, file_type="subh", snapnum=snapshot, units=True, comoving=True
    )
    h = group_info.h
    z = group_info.z

    if "FOF" not in mass_dset:
        raise ValueError("group_dset should be a FOF property")
    if "FOF" not in coord_dset:
        raise ValueError("coord_dset should be a FOF property")
    if not np.all(["FOF" in extra_dset for extra_dset in extra_dsets]):
        raise ValueError("extra_dsets should be FOF properties")

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(group_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fname = (save_dir / f"{info_fname}_{snapshot:03d}").with_suffix(".hdf5")

    coordinates = group_info.read_var(coord_dset, verbose=verbose)
    radii = group_info.read_var(radius_dset, verbose=verbose)
    masses = group_info.read_var(mass_dset, verbose=verbose)
    group_ids = np.arange(len(masses))

    mass_range = mass_range.to(masses.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    mass_selection = (masses > np.min(mass_range)) & (masses < np.max(mass_range))

    # also select coordinate range
    if coord_range is not None:
        coord_range = coord_range.to(coordinates.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
        coord_selection = np.all(
            [
                (coordinates[:, i] > np.min(coord_range[i])) & (coordinates[:, i] < np.max(coord_range[i]))
                for i in range(coord_range.shape[0])
            ], axis=0
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
            data=np.arange(0, masses.shape[0]),
            index=masses,
            bin_edges=mass_bin_edges,
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

    coordinates = coordinates[selection]
    radii = radii[selection]
    masses = masses[selection]
    group_ids = group_ids[selection].astype(int)

    if logger:
        logger.info(f"coordinates and masses sliced")

    extra = {
        extra_dset: group_info.read_var(extra_dset, verbose=verbose)[selection]
        for extra_dset in (extra_dsets or {})
    }

    data = {
        "attrs": {
            "description": "File with selected coordinates for maps.",
        },
        "coordinates": coordinates,
        "radii": radii,
        "masses": masses,
        "group_ids": group_ids,
        "z": z,
        "radius_dset": radius_dset,
        "mass_dset": mass_dset,
        "mass_range": mass_range,
        **extra,
    }

    if coord_range is not None:
        data["coord_range"] = coord_range

    io.dict_to_hdf5(fname=fname, data=data, overwrite=True)
    if logger:
        logger.info(f"Finished saving coordinates for {sim_dir=} and {snapshot=}")

    return str(fname)
