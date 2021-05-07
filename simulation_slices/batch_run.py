import os
from pathlib import Path
from typing import List

import h5py
import numpy as np

from simulation_slices import Config
import simulation_slices.maps.analysis as analysis
import simulation_slices.maps.generation as map_gen
import simulation_slices.sims.bahamas as bahamas
import simulation_slices.sims.mira_titan as mira_titan


def save_coords(sim_idx: int, config: Config) -> List[str]:
    """Save a set of halo centers to generate maps around."""
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    snapshots = config.snapshots[sim_idx]
    box_size = config.box_sizes[sim_idx]
    save_dir = config.slice_paths[sim_idx]

    mass_dset = config.mass_dset
    mass_range = config.mass_range
    coord_dset = config.coord_dset
    extra_dsets = config.extra_dsets
    save_dir = config.coords_paths[sim_idx]
    coords_fname = config.coords_name

    all_fnames = []
    if sim_suite.lower() == "bahamas":
        for snap in np.atleast_1d(snapshots):
            fname = bahamas.save_coords_file(
                sim_dir=str(sim_dir),
                snapshot=snap,
                mass_dset=mass_dset,
                coord_dset=coord_dset,
                mass_range=mass_range,
                extra_dsets=extra_dsets,
                save_dir=save_dir,
                coords_fname=coords_fname,
                verbose=False,
            )
            all_fnames = [*all_fnames, fname]

    elif sim_suite.lower() == "miratitan":
        for snap in np.atleast_1d(snapshots):
            fname = mira_titan.save_coords_file(
                sim_dir=str(sim_dir),
                box_size=box_size,
                snapshot=snap,
                mass_range=mass_range,
                save_dir=save_dir,
                coords_fname=coords_fname,
            )
            all_fnames = [*all_fnames, fname]

    return all_fnames


def slice_sim(sim_idx: int, config: Config) -> List[str]:
    """Save a set of slices for sim_idx in config.sim_paths."""
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    snapshots = config.snapshots[sim_idx]
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]
    save_dir = config.slice_paths[sim_idx]

    slice_axes = config.slice_axes
    slice_size = config.slice_size

    all_fnames = []
    if sim_suite.lower() == "bahamas":
        for snap in np.atleast_1d(snapshots):
            fnames = bahamas.save_slice_data(
                sim_dir=str(sim_dir),
                snapshot=snap,
                ptypes=ptypes,
                slice_axes=slice_axes,
                slice_size=slice_size,
                save_dir=save_dir,
                verbose=False,
            )
            all_fnames = [*all_fnames, *(fnames or [])]

    elif sim_suite.lower() == "miratitan":
        for snap in np.atleast_1d(snapshots):
            fnames = mira_titan.save_slice_data(
                sim_dir=str(sim_dir),
                snapshot=snap,
                box_size=box_size,
                slice_axes=slice_axes,
                slice_size=slice_size,
                save_dir=save_dir,
                verbose=False,
            )
            all_fnames = [*all_fnames, *(fnames or [])]

    return all_fnames


def map_coords(sim_idx: int, config: Config) -> List[str]:
    """Save a set of maps for sim_idx in config.sim_paths."""
    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    snapshots = config.snapshots[sim_idx]
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]

    coords_name = config.coords_name
    coords_file = config.coords_files[sim_idx]
    all_fnames = []

    if config.compute_coords:
        fnames = save_coords(sim_idx=sim_idx, config=config)
        all_fnames = [*all_fnames, *(fnames or [])]

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
        fnames = map_gen.save_maps(
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
        all_fnames = [*all_fnames, *(fnames or [])]

    return all_fnames


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
