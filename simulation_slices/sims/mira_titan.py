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
import simulation_slices.maps.generation as map_gen
import simulation_slices.maps.map_layout as map_layout
import simulation_slices.maps.observables as obs
import simulation_slices.sims.slicing as slicing
import simulation_slices.utilities as util


PROPS_PTYPES = {"coordinates": f"dm/coordinates", "masses": f"dm/masses"}


def save_coords_file(
    sim_dir: str,
    box_size: u.Quantity,
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
    box_size : astropy.units.Quantity
        size of simulation
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
        box_size=box_size,
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
    coordinates = (
        np.vstack(
            [
                group_data["sod_halo_min_pot_x"],
                group_data["sod_halo_min_pot_y"],
                group_data["sod_halo_min_pot_z"],
            ]
        )
        .T
        .to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    )
    group_ids = group_data["fof_halo_tag"]
    masses = group_data["sod_halo_mass"]

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
        sampled_ids = util.groupby(np.arange(0, masses.shape[0]), masses, mass_bin_edges)

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

    layout = {
        "attrs": {
            "description": "File with selected coordinates for maps.",
        },
        "dsets": {
            "coordinates": {
                "data": coordinates,
                "attrs": {
                    "description": "Centers",
                    "units": "Mpc",
                    "mass_range": mass_range.value,
                    "mass_range_units": str(mass_range.unit),
                },
            },
            "group_ids": {
                "data": group_ids,
                "attrs": {
                    "description": "Group IDs",
                },
            },
            "masses": {
                "data": masses,
                "attrs": {
                    "description": "Spherical overdensity masses m200c",
                    "units": "Msun",
                },
            },
        },
    }

    if coord_range is not None:
        layout["dsets"]["coordinates"]["attrs"]["coord_range"] = coord_range.value
        layout["dsets"]["coordinates"]["attrs"]["coord_range_units"] = str(coord_range.unit)

    io.create_hdf5(fname=fname, layout=layout, overwrite=True, close=True)
    return str(fname)


def save_subvolumes(
    sim_dir: str,
    box_size: u.Quantity,
    snapshot: int,
    mass_range: Tuple[u.Quantity, u.Quantity],
    coord_ranges: u.Quantity = None,
    save_dir: Optional[str] = None,
    coords_fnames: Optional[str] = "",
    logger: util.LoggerType = None,
    **kwargs,
) -> str:
    """For snapshot of simulation in sim_dir, save coordinates of haloes
    within group_range.

    Parameters
    ----------
    sim_dir : str
        path of the simulation
    box_size : astropy.units.Quantity
        size of simulation
    snapshot : int
        snapshot to look at
    mass_range : (min, max) tuple
        minimum and maximum value for masses
    coord_ranges : (n, 3, 2) array
        ranges for coordinates to include
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/maps/
    coords_fnames : (n,) list of strings
        names for the coordinates files without extension

    Returns
    -------
    fname : str
        filename of coords file
    saves a set of coordinates to save_dir

    """
    sim_info = MiraTitan(
        sim_dir=sim_dir,
        box_size=box_size,
        snapnum=snapshot,
        verbose=False,
    )
    h = sim_info.cosmo["h"]

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    tl0 = time.time()
    group_data = sim_info.read_properties(
        "fof",
        [
            "fof_halo_mass",
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "fof_halo_tag",
        ],
    )
    tl1 = time.time()
    if logger:
        logger.debug(f"loading fof data took {tl1 - tl0:.2f}s")

    coordinates = (
        np.vstack(
            [
                group_data["fof_halo_center_x"],
                group_data["fof_halo_center_y"],
                group_data["fof_halo_center_z"],
            ]
        )
        .T
        .to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    )
    masses = group_data["fof_halo_mass"].to(
        "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    group_ids = group_data["fof_halo_tag"].astype(int)

    mass_range = mass_range.to(masses.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    mass_selection = (masses > np.min(mass_range)) & (masses < np.max(mass_range))

    fnames = []
    # also select coordinate range
    for idx, coord_range in enumerate(coord_ranges):
        tc0 = time.time()
        fname = (save_dir / f"{coords_fnames[idx]}_{snapshot:03d}").with_suffix(".hdf5")
        coord_range = coord_range.to(coordinates.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
        coord_selection = np.all(
            [
                (coordinates[:, i] > np.min(coord_range[i])) & (coordinates[:, i] < np.max(coord_range[i]))
                for i in range(coord_range.shape[0])
            ], axis=0
        )
        selection = mass_selection & coord_selection

        layout = {
            "attrs": {
                "description": "File with selected coordinates for maps.",
            },
            "dsets": {
                "coordinates": {
                    "data": coordinates[selection],
                    "attrs": {
                        "description": "Centers",
                        "units": str(coordinates.unit),
                        "mass_range": mass_range.value,
                        "mass_range_units": str(mass_range.unit),
                        "coord_range": coord_range.value,
                        "coord_range_units": str(coord_range.unit),
                    },
                },
                "group_ids": {
                    "data": group_ids[selection],
                    "attrs": {
                        "description": "Group IDs",
                    },
                },
                "masses": {
                    "data": masses[selection],
                    "attrs": {
                        "description": "Masses",
                        "units": str(masses.unit),
                    },
                },
            },
        }

        io.create_hdf5(fname=fname, layout=layout, close=True)
        fnames.append(str(fname))
        tc1 = time.time()
        if logger:
            logger.debug("saving coord_range took {tc1 - tc0:.2f}s")

    return fnames


def save_slice_data(
    sim_dir: str,
    box_size: u.Quantity,
    snapshot: int,
    slice_axes: List[int] = [0, 1, 2],
    num_slices: int = 1000,
    save_dir: Optional[str] = None,
    logger: util.LoggerType = None,
    verbose: Optional[bool] = False,
    **kwargs,
) -> List[str]:
    """For snapshot of simulation in sim_dir, slice the particle data for
    all ptypes along the x, y, and z directions. Slices are saved
    in the Snapshots directory by default.

    Parameters
    ----------
    sim_dir : str
        path of simulation
    box_size : astropy.units.Quantity
        size of simulation, in Mpc for MiraTitan
    snapshot : int
        snapshot to look at
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    num_slices : int
        total number of slices
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/slices/
    verbose : bool
        print progress bar

    Returns
    -------
    fnames : list of str
    saves particles for each slice in the snapshot_xxx/slices/
    directory

    """
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
        save_dir = util.check_path(sim_info.get_fname("snap")).parent / "slices"
    else:
        save_dir = util.check_path(save_dir)

    # crude estimate of maximum number of particles in each slice
    N_tot = sim_info.num_part_tot
    maxshape = int(2 * N_tot / num_slices)

    fnames = []
    for slice_axis in slice_axes:
        # create the hdf5 file to fill up
        fname = slicing.create_slice_file(
            save_dir=save_dir,
            snapshot=snapshot,
            box_size=sim_info.box_size,
            z=sim_info.z,
            a=sim_info.a,
            h=h,
            ptypes=["dm"],
            num_slices=num_slices,
            slice_axis=slice_axis,
            maxshape=maxshape,
        )
        fnames.append(fname)

    # now loop over all snapshot files and add their particle info
    # to the correct slice

    if verbose:
        num_files_range = tqdm(
            sim_info.datatype_info["snap"]["nums"], desc="Slicing particle files"
        )
    else:
        num_files_range = sim_info.datatype_info["snap"]["nums"]

    for file_num in num_files_range:
        properties = sim_info.read_properties(
            datatype="snap", props=["x", "y", "z"], num=file_num
        )

        # MiraTitan box size is in Mpc, cannot be converted in Config
        # need to enforce consistent units => get rid of all littleh factors
        coords = np.vstack([properties["x"], properties["y"], properties["z"]]).to(
            "Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
        )
        masses = np.atleast_1d(sim_info.simulation_info["snap"]["m_p"]).to(
            "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
        )

        properties = {"coordinates": coords, "masses": masses}
        # write each slice to a separate file
        for slice_axis in slice_axes:
            slice_dict = slicing.slice_particle_list(
                box_size=box_size,
                num_slices=num_slices,
                slice_axis=slice_axis,
                properties=properties,
            )

            fname = slicing.slice_file_name(
                save_dir=save_dir,
                slice_axis=slice_axis,
                num_slices=num_slices,
                snapshot=snapshot,
            )
            h5file = h5py.File(fname, "r+")

            # append results to hdf5 file
            for idx, (coord, masses) in enumerate(
                zip(slice_dict["coordinates"], slice_dict["masses"])
            ):
                if not coord:
                    continue

                coord_dset = f'{idx}/{PROPS_PTYPES["coordinates"]}'
                io.add_to_hdf5(
                    h5file=h5file,
                    dataset=coord_dset,
                    vals=coord[0],
                    axis=1,
                )

                # only want to add single value for dm mass
                if h5file[f'{idx}/{PROPS_PTYPES["masses"]}'].shape[0] == 0:
                    mass_dset = f'{idx}/{PROPS_PTYPES["masses"]}'
                    io.add_to_hdf5(
                        h5file=h5file,
                        dataset=mass_dset,
                        vals=np.unique(masses[0]),
                        axis=0,
                    )

            h5file.close()

    return fnames


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
    properties = sim_info.read_properties(
        datatype="snap",
        props=["x", "y", "z"]
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
            **properties
        )
        map_files[slice_axis]["map_file"]["dm_mass"][()] = mp.value
        map_files[slice_axis]["map_file"]["dm_mass"].attrs["units"] = str(mp.unit)

        ts1 = time.time()
        if logger:
            logger.info(
                f"{slice_axis=} finished in {ts1 - ts0:.2f}s"
            )

    # finished file_num
    tf = time.time()
    if logger:
        logger.info(
            f"{slice_axes=} finished in {tf - ts:.2f}s"
        )

    t1 = time.time()
    if logger:
        logger.info(f"Finished {slice_axes=} for {sim_dir=} took {t1 - t0:.2f}s")

    # need to close map_files
    for slice_axis in slice_axes:
        map_files[slice_axis]["map_file"].close()

    return fnames
