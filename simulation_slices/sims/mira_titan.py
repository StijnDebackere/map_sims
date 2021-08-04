import logging
import time
from typing import List, Optional, Tuple
from pathlib import Path

import astropy.units as u
import h5py
from mira_titan import MiraTitan
import numpy as np
from tqdm import tqdm

import simulation_slices.io as io
import simulation_slices.utilities as util


def get_file_nums(
    sim_dir: str,
    snapshot: int,
) -> List[int]:
    sim_info = MiraTitan(
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
):
    prop_options = ["coordinates", "masses"]

    if properties is None:
        return {}

    valid_props = set(prop_options) & set(properties)
    if not valid_props:
        raise ValueError(f"properties should be in {prop_options=}")

    sim_info = MiraTitan(
        sim_dir=sim_dir,
        snapnum=snapshot,
        verbose=verbose,
    )
    h = sim_info.cosmo["h"]

    props = {}
    if "coordinates" in valid_props:
        data = sim_info.read_properties(
            datatype="snap",
            props=["x", "y", "z"],
            num=file_num,
        )
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

    sim_info = MiraTitan(
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


def save_halo_coords_file(
    sim_dir: str,
    snapshot: int,
    mass_range: Tuple[u.Quantity, u.Quantity],
    coord_range: u.Quantity = None,
    save_dir: Optional[str] = None,
    coords_fname: Optional[str] = "",
    sample_haloes_bins: Optional[dict] = None,
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
    coords_fname : str
        name for the coordinates file without extension

    Returns
    -------
    fname : str
        filename of coords file
    saves a set of coordinates to save_dir

    """
    sim_info = MiraTitan(
        sim_dir=sim_dir,
        snapnum=snapshot,
        verbose=False,
    )
    h = sim_info.cosmo["h"]

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fname = (save_dir / f"{coords_fname}_{snapshot:03d}").with_suffix(".hdf5")

    group_data = sim_info.read_properties(
        "sod",
        [
            "sod_halo_mass",
            "sod_halo_min_pot_x",
            "sod_halo_min_pot_y",
            "sod_halo_min_pot_z",
            "fof_halo_tag",
        ],
    )
    coordinates = np.vstack(
        [
            group_data["sod_halo_min_pot_x"],
            group_data["sod_halo_min_pot_y"],
            group_data["sod_halo_min_pot_z"],
        ]
    ).T.to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    group_ids = group_data["fof_halo_tag"]
    masses = group_data["sod_halo_mass"]

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

    # only included selection
    coordinates = coordinates[selection]
    masses = masses[selection].to(
        "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    group_ids = group_ids[selection].astype(int)

    data = {
        "attrs": {
            "description": "File with selected coordinates for maps.",
        },
        "coordinates": coordinates,
        "mass_dset": mass_dset,
        "mass_range": mass_range,
        "group_ids": group_ids,
        "masses": masses,
    }

    if coord_range is not None:
        data["coord_range"] = coord_range

    io.dict_to_hdf5(fname=fname, data=data, overwrite=True)
    return str(fname)


def save_full_maps(
    sim_dir: str,
    snapshot: int,
    slice_axes: int,
    box_size: u.Quantity,
    map_pix: int,
    save_dir: str,
    map_name_append: str = "",
    downsample: bool = False,
    downsample_factor: float = None,
    overwrite: bool = False,
    swmr: bool = False,
    method: str = None,
    n_ngb: int = 30,
    verbose: bool = False,
    logger: util.LoggerType = None,
    **kwargs,
) -> List[str]:
    """Project full simulation in a map of (map_pix, map_pix) for slice_axes.

    Parameters
    ----------
    sim_dir : str
        directory of the simulation
    snapshot : int
        snapshot to look at
    slice_axes : int
        axis to slice along [x=0, y=1, z=2]
    box_size : astropy.units.Quantity
        size of simulation
    map_pix : int
        square root of number of pixels in map
    save_dir : str
        directory to save map files to
    map_name_append : str
        optional extra to append to filenames
    overwrite : bool
        overwrite map_file if already exists
    swmr : bool
        enable single writer multiple reader mode for map_file
    method : str ["sph", "bin"]
        method for map projection: sph smoothing with n_ngb neighbours or 2D histogram
    n_ngb : int
        number of neighbours to determine SPH kernel size
    verbose : bool
        show progress bar

    Returns
    -------
    saves maps to {save_dir}/{slice_axis}_maps_{coords_name}{map_name_append}_{snapshot:03d}.hdf5

    """
    t0 = time.time()
    slice_axes = np.atleast_1d(slice_axes)
    sim_info = MiraTitan(
        sim_dir=sim_dir,
        box_size=box_size,
        snapnum=snapshot,
        verbose=verbose,
    )
    # read in the Mpc unit box_size
    box_size = sim_info.L
    h = sim_info.cosmo["h"]

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.get_fname("snap")).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fnames = []
    map_files = {}
    for slice_axis in slice_axes:
        map_name = map_gen.get_map_name(
            save_dir=save_dir,
            slice_axis=slice_axis,
            snapshot=snapshot,
            method=method,
            map_thickness=box_size,
            coords_name="",
            map_name_append=map_name_append,
            downsample=downsample,
            downsample_factor=downsample_factor,
            full=True,
        )
        map_file = map_layout.create_map_file(
            map_name=map_name,
            overwrite=overwrite,
            close=False,
            # cannot have swmr since we are adding attributes later
            swmr=False,
            slice_axis=slice_axis,
            box_size=box_size,
            map_types=["dm_mass"],
            map_size=box_size,
            map_thickness=box_size,
            map_pix=map_pix,
            snapshot=snapshot,
            n_ngb=n_ngb,
            maxshape=0,
            full=True,
        )
        map_files[slice_axis] = {
            "map_file": map_file,
        }
        fnames.append(map_name)

    ts = time.time()
    properties = sim_info.read_properties(datatype="snap", props=["x", "y", "z"])
    tr = time.time()
    if logger:
        logger.info(f"properties read in {tr - ts:.2f}s")

    # MiraTitan box size is in Mpc, cannot be converted in Config
    # need to enforce consistent units => get rid of all littleh factors
    coords = np.vstack([properties["x"], properties["y"], properties["z"]]).to(
        "Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    masses = np.atleast_1d(sim_info.simulation_info["snap"]["m_p"]).to(
        "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )

    properties = {"masses": masses}

    # write each slice to a separate file
    for slice_axis in slice_axes:
        ts0 = time.time()
        no_slice_axis = np.arange(0, 3) != slice_axis
        if method == "bin":
            coords_to_map = map_gen.coords_to_map_bin
        elif method == "sph":
            coords_to_map = map_gen.coords_to_map_sph
            properties = {**properties, "n_ngb": n_ngb}

        mp = coords_to_map(
            coords=coords[no_slice_axis],
            map_size=box_size,
            map_pix=map_pix,
            box_size=box_size,
            func=obs.particles_masses,
            map_center=None,
            logger=logger,
            **properties,
        )
        map_files[slice_axis]["map_file"]["dm_mass"][()] = mp.value
        map_files[slice_axis]["map_file"]["dm_mass"].attrs["units"] = str(mp.unit)

        ts1 = time.time()
        if logger:
            logger.info(f"{slice_axis=} finished in {ts1 - ts0:.2f}s")

    # finished file_num
    t1 = time.time()
    if logger:
        logger.info(f"Finished {slice_axes=} for {sim_dir=} took {t1 - t0:.2f}s")

    # need to close map_files
    for slice_axis in slice_axes:
        map_files[slice_axis]["map_file"].close()

    return fnames


def save_maps_los(
    sim_dir: str,
    snapshot: int,
    slice_axis: int,
    box_size: u.Quantity,
    map_pix: int,
    map_thickness: u.Quantity,
    save_dir: str,
    map_name_append: str = "",
    downsample: bool = False,
    downsample_factor: float = None,
    overwrite: bool = False,
    swmr: bool = False,
    method: str = None,
    n_ngb: int = 30,
    verbose: bool = False,
    logger: util.LoggerType = None,
    **kwargs,
) -> List[str]:
    """Project full simulation in a map of (map_pix, map_pix) for
    slice_axes for different map_thicknesses.

    Parameters
    ----------
    sim_dir : str
        directory of the simulation
    snapshot : int
        snapshot to look at
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    box_size : astropy.units.Quantity
        size of simulation
    map_pix : int
        square root of number of pixels in map
    map_thickness : astropy.units.Quantity
        thickness of the map projection
    save_dir : str
        directory to save map files to
    map_name_append : str
        optional extra to append to filenames
    overwrite : bool
        overwrite map_file if already exists
    swmr : bool
        enable single writer multiple reader mode for map_file
    method : str ["sph", "bin"]
        method for map projection: sph smoothing with n_ngb neighbours or 2D histogram
    n_ngb : int
        number of neighbours to determine SPH kernel size
    verbose : bool
        show progress bar

    Returns
    -------
    saves maps to {save_dir}/{slice_axis}_maps_full{map_name_append}_{snapshot:03d}.hdf5

    """
    t0 = time.time()
    sim_info = MiraTitan(
        sim_dir=sim_dir,
        box_size=box_size,
        snapnum=snapshot,
        verbose=verbose,
    )
    # read in the Mpc unit box_size
    box_size = sim_info.L
    h = sim_info.cosmo["h"]

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.get_fname("snap")).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    # go from thick to thin
    map_thickness = np.sort(map_thickness)[::-1]

    map_name = map_gen.get_map_name(
        save_dir=save_dir,
        slice_axis=slice_axis,
        snapshot=snapshot,
        method=method,
        map_thickness=map_thickness,
        coords_name="",
        map_name_append=map_name_append,
        downsample=downsample,
        downsample_factor=downsample_factor,
        full=True,
    )
    map_file = map_layout.create_map_file(
        map_name=map_name,
        overwrite=overwrite,
        close=False,
        # cannot have swmr since we are adding attributes later
        swmr=False,
        slice_axis=slice_axis,
        box_size=box_size,
        map_types=["dm_mass"],
        map_size=box_size,
        map_thickness=map_thickness,
        map_pix=map_pix,
        snapshot=snapshot,
        n_ngb=n_ngb,
        maxshape=0,
        full=True,
    )

    ts = time.time()
    properties = sim_info.read_properties(
        datatype="snap",
        props=["x", "y", "z"],
    )
    tr = time.time()
    if logger:
        logger.info(f"properties read in {tr - ts:.2f}s")

    # MiraTitan box size is in Mpc, cannot be converted in Config
    # need to enforce consistent units => get rid of all littleh factors
    coords = np.vstack([properties["x"], properties["y"], properties["z"]]).to(
        "Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    masses = np.atleast_1d(sim_info.simulation_info["snap"]["m_p"]).to(
        "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )

    properties = {"masses": masses}

    no_slice_axis = np.arange(0, 3) != slice_axis
    # we will gradually slice down the list of coordinates to only contain
    # particles within box_size / 2 +/- dl / 2 along slice_axis
    coords_slice = np.copy(coords)
    for idx_l, dl in enumerate(map_thickness):
        ts0 = time.time()
        if dl >= box_size:
            coords_slice = coords_slice
        elif dl < box_size:
            slice_sel = (
                coords_slice[slice_axis] >= 0.5 * (box_size - map_thickness)
            ) & ((coords_slice[slice_axis] <= 0.5 * (box_size + map_thickness)))
            coords_slice = coords_slice[:, slice_sel]

        if logger:
            ts1 = time.time()
            logger.info("{dl=} slicing took {ts1 - ts0:.2f}s")

        if method == "bin":
            coords_to_map = map_gen.coords_to_map_bin
        elif method == "sph":
            coords_to_map = map_gen.coords_to_map_sph
            properties = {**properties, "n_ngb": n_ngb}

        mp = coords_to_map(
            coords=coords_slice[no_slice_axis],
            map_size=box_size,
            map_pix=map_pix,
            box_size=box_size,
            func=obs.particles_masses,
            map_center=None,
            logger=logger,
            **properties,
        )
        map_file["dm_mass"][..., idx_l] = mp.value
        map_file["dm_mass"].attrs["units"] = str(mp.unit)

        if logger:
            ts1 = time.time()
            logger.info(f"{dl=} finished in {ts1 - ts0:.2f}s")

    t1 = time.time()
    if logger:
        logger.info(
            f"Finished {slice_axis=}, {map_thickness=} and {snapshot=} for {sim_dir=} in {t1 - t0:.2f}s"
        )

    fname = map_file.filename
    # still need to close the HDF5 files
    map_file.close()

    return fname
