import os
from pathlib import Path

from gadget import Gadget
import h5py
import numpy as np
import toml

from simulation_slices import Config
from simulation_slices.parallel import compute_tasks
import simulation_slices.maps.analysis as analysis
import simulation_slices.maps.generation as map_gen
import simulation_slices.sims.bahamas as bahamas
import simulation_slices.utilities as util

import pdb


def order_coords(coords, map_thickness, box_size, slice_axis):
    """Order the list of coords such that each cpu accesses independent
    slice_files.

    Parameters
    ----------
    coords : (3, N) array
        coordinates to order
    map_thickness : float
        thickness of the map matching units of box_size
    box_size : float
        size of the box
    slice_axis : int
        coordinate to slice along

    Returns
    -------
    coords_split : list of coords
        coordinates split up in box_size / map_thickness bins
    """
    # divide the box up in independent regions of map_thickness
    bin_edges = np.arange(0, box_size, map_thickness)

    # sort the coords according to slice_axis
    coords_sorted = coords[:, coords[slice_axis].argsort()]
    bin_ids = np.digitize(coords[slice_axis], bin_edges)
    in_bins = np.unique(bin_ids)

    coords_split = [coords[:, bin_ids == idx] for idx in in_bins]
    return coords_split


def save_coords(
        sim_dir, sim_type, snapshots, group_dset, coord_dset, group_range, extra_dsets,
        save_dir, coords_fname):
    if sim_type == 'BAHAMAS':
        for snap in np.atleast_1d(snapshots):
            bahamas.save_coords_file(
                base_dir=str(sim_dir), snapshot=snap, group_dset=group_dset,
                coord_dset=coord_dset, group_range=group_range, extra_dsets=extra_dsets,
                save_dir=save_dir, coords_fname=coords_fname, verbose=False
            )

    return (os.getpid(), f'{save_dir} coords saved')


def slice_sim(sim_dir, sim_type, snapshots, ptypes, slice_axes, slice_size, save_dir):
    if sim_type == 'BAHAMAS':
        for snap in np.atleast_1d(snapshots):
            bahamas.save_slice_data(
                base_dir=str(sim_dir), snapshot=snap, ptypes=ptypes,
                slice_axes=slice_axes, slice_size=slice_size,
                save_dir=save_dir, verbose=False
            )

    return (os.getpid(), f'{sim_dir} sliced')


def map_coords(
        snapshots, box_size, coords_file, coords_name,
        slice_dir, slice_axes, slice_size,
        map_types, map_size, map_res, map_thickness, save_dir):
    with h5py.File(str(coords_file), 'r') as h5file:
        centers = h5file['coordinates'][:]

    for snap in np.atleast_1d(snapshots):
        maps = map_gen.save_maps(
            centers=centers, slice_dir=slice_dir, snapshot=snap,
            slice_axes=slice_axes, slice_size=slice_size, box_size=box_size,
            map_size=map_size, map_res=map_res, map_thickness=map_thickness,
            map_types=map_types, save_dir=save_dir, coords_name=coords_name,
        )

    return (os.getpid(), f'{save_dir} maps saved')


def analyze_map():
    pass


def slice_sims(config, n_workers):
    for p in config.slice_paths:
        p.mkdir(parents=True, exist_ok=True)

    kwargs_list = []
    for sim_dir, snaps, ptypes, save_dir in zip(
            config.sim_paths, config.snapshots,
            config.ptypes, config.slice_paths):
        kwargs_list.append(dict(
            sim_dir=sim_dir,
            sim_type=config.sim_type,
            ptypes=ptypes,
            snapshots=snaps,
            slice_axes=config.slice_axes,
            slice_size=config.slice_size,
            save_dir=save_dir
        ))

    result_slices = compute_tasks(slice_sim, n_workers, kwargs_list)


def compute_maps(config, n_workers):
    if config.compute_coords:
        kwargs_list = []
        for sim_dir, save_dir in zip(config.sim_paths, config.coords_paths):
            kwargs_list.append(dict(
                sim_dir=sim_dir,
                sim_type=config.sim_type,
                snapshots=config.snapshots,
                group_dset=config.group_dset,
                coord_dset=config.coord_dset,
                group_range=config.group_range,
                extra_dsets=config.extra_dsets,
                save_dir=save_dir,
                coords_fname=config.coords_name,
            ))

        result_coords = compute_tasks(save_coords, n_workers, kwargs_list)

    kwargs_list = []
    for map_dir, slice_dir, coords_file, box_size in zip(
            config.map_paths, config.slice_paths, config.coords_files, config.box_sizes):
        kwargs_list.append(dict(
            snapshots=config.snapshots,
            box_size=box_size,
            coords_file=coords_file,
            coords_name=config.coords_name,
            slice_dir=slice_dir,
            slice_axes=config.slice_axes,
            slice_size=config.slice_size,
            map_types=config.map_types,
            map_size=config.map_size,
            map_res=config.map_res,
            map_thickness=config.map_thickness,
            save_dir=map_dir,
        ))

    result_maps = compute_tasks(map_coords, n_workers, kwargs_list)


def run_pipeline(
        config_file, n_workers=10,
        sims=True, maps=True, observables=True):
    config = Config(config_file)

    if sims:
        slice_sims(config, n_workers)

    if maps:
        compute_maps(config, n_workers)

    if observables:
        pass
