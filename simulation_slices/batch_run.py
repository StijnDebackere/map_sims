import os

import h5py
import numpy as np

from simulation_slices import Config
from simulation_slices.parallel import compute_tasks
import simulation_slices.maps.analysis as analysis
import simulation_slices.maps.generation as map_gen
import simulation_slices.sims.bahamas as bahamas
import simulation_slices.sims.mira_titan as mira_titan


def save_coords(
    base_dir,
    sim_dir,
    snapshots,
    sim_suite,
    box_size,
    group_dset,
    coord_dset,
    group_range,
    extra_dsets,
    save_dir,
    coords_fname,
):
    """Save a set of halo centers to generate maps around."""
    if sim_suite.lower() == "bahamas":
        for snap in np.atleast_1d(snapshots):
            bahamas.save_coords_file(
                sim_dir=str(sim_dir),
                snapshot=snap,
                group_dset=group_dset,
                coord_dset=coord_dset,
                group_range=group_range,
                extra_dsets=extra_dsets,
                save_dir=save_dir,
                coords_fname=coords_fname,
                verbose=False,
            )

    elif sim_suite.lower() == "miratitan":
        for snap in np.atleast_1d(snapshots):
            mira_titan.save_coords_file(
                base_dir=str(base_dir),
                sim_dir=str(sim_dir),
                box_size=box_size,
                snapshot=snap,
                group_range=group_range,
                save_dir=save_dir,
                coords_fname=coords_fname,
            )

    return (os.getpid(), f"{save_dir} coords saved")


def slice_sim(sim_idx, config):
    """Save a set of slices for sim_idx in config.sim_paths."""
    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    snapshots = config.snapshots[sim_idx]
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]
    save_dir = config.slice_paths[sim_idx]

    slice_axes = config.slice_axes
    slice_size = config.slice_size
    if sim_suite.lower() == "bahamas":
        for snap in np.atleast_1d(snapshots):
            bahamas.save_slice_data(
                sim_dir=str(sim_dir),
                snapshot=snap,
                ptypes=ptypes,
                slice_axes=slice_axes,
                slice_size=slice_size,
                save_dir=save_dir,
                verbose=False,
            )
    elif sim_suite.lower() == "miratitan":
        for snap in np.atleast_1d(snapshots):
            mira_titan.save_slice_data(
                base_dir=str(sim_dir),
                snapshot=snap,
                box_size=box_size,
                slice_axes=slice_axes,
                slice_size=slice_size,
                save_dir=save_dir,
                verbose=False,
            )

    return (os.getpid(), f"{sim_dir} sliced")


def map_coords(sim_idx, config):
    """Save a set of maps for sim_idx in config.sim_paths."""
    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    snapshots = config.snapshots[sim_idx]
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]

    coords_file = config.coords_files[sim_idx]
    coords_name = config.coords_name
    if config.compute_coords:
        coords_dir = config.coords_paths[sim_idx]
        save_coords(
            base_dir=base_dir,
            sim_dir=sim_dir,
            sim_suite=sim_suite,
            snapshots=snapshots,
            box_size=box_size,
            group_dset=config.group_dset,
            coord_dset=config.coord_dset,
            group_range=config.group_range,
            extra_dsets=config.extra_dsets,
            save_dir=coords_dir,
            coords_fname=config.coords_name,
        )

    slice_dir = config.slice_paths[sim_idx]
    slice_axes = config.slice_axes
    slice_size = config.slice_size

    save_dir = config.map_paths[sim_idx]
    map_types = config.map_types[sim_idx]
    map_size = config.map_size
    map_res = config.map_res
    map_thickness = config.map_thickness
    with h5py.File(str(coords_file), "r") as h5file:
        centers = h5file["coordinates"][:]

    for snap in np.atleast_1d(snapshots):
        map_gen.save_maps(
            centers=centers,
            slice_dir=slice_dir,
            snapshot=snap,
            slice_axes=slice_axes,
            slice_size=slice_size,
            box_size=box_size,
            map_size=map_size,
            map_res=map_res,
            map_thickness=map_thickness,
            map_types=map_types,
            save_dir=save_dir,
            coords_name=coords_name,
        )

    return (os.getpid(), f"{save_dir} maps saved")


def analyze_map(
    snapshots,
    box_size,
    coords_name,
    slice_dir,
    slice_axes,
    slice_size,
    obs_info,
    save_dir,
):
    for obs_type in obs_info["obs_types"]:
        for map_type in obs_info[obs_type]["map_types"]:
            analysis.save_observable()
