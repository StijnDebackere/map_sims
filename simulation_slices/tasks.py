import logging
import os
from pathlib import Path
import time
from typing import List, Optional, Union

import astropy.units as u
import h5py
import numpy as np

from simulation_slices import Config
import simulation_slices.utilities as util
import simulation_slices.maps.analysis as analysis
import simulation_slices.maps.generation as map_gen
import simulation_slices.sims.bahamas as bahamas
import simulation_slices.sims.mira_titan as mira_titan


def get_logger(sim_idx: int, config: Config, fname: str) -> logging.Logger:
    logger = logging.getLogger(f"{os.getpid()} - batch_run")

    log_dir = config.log_dir
    # save to log file
    fh = logging.FileHandler(
        f"{log_dir}/{fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.log"
    )

    if config.log_level.lower() == "info":
        fh.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    elif config.log_level.lower() == "debug":
        fh.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif config.log_level.lower() == "warning":
        fh.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)
    elif config.log_level.lower() == "critical":
        fh.setLevel(logging.CRITICAL)
        logger.setLevel(logging.CRITICAL)

    # set formatting
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s [%(levelname)s] %(funcName)s - %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def slice_sim(
    sim_idx: int,
    snapshot: int,
    config: Union[Config, str],
    logger: util.LoggerType = None,
) -> List[str]:
    """Save a set of slices for sim_idx in config.sim_paths."""
    start = time.time()

    if type(config) is str:
        config = Config(config)

    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]
    save_dir = config.slice_paths[sim_idx]

    slice_axes = config.slice_axes
    num_slices = config.num_slices

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            config=config,
            fname=f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_slice_sim{config.log_name_append}",
        )

    if sim_suite.lower() == "bahamas":
        fnames = bahamas.save_slice_data(
            sim_dir=str(sim_dir),
            snapshot=snap,
            ptypes=ptypes,
            slice_axes=slice_axes,
            num_slices=num_slices,
            save_dir=save_dir,
            verbose=False,
            logger=logger,
        )

    elif sim_suite.lower() == "miratitan":
        fnames = mira_titan.save_slice_data(
            sim_dir=str(sim_dir),
            box_size=box_size,
            snapshot=snapshot,
            slice_axes=slice_axes,
            num_slices=num_slices,
            save_dir=save_dir,
            verbose=False,
            logger=logger,
        )

    end = time.time()
    if logger:
        logger.info(
            f"slice_sim_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
    # return fnames


def save_coords(
    sim_idx: int,
    snapshot: int,
    config: Union[Config, str],
    logger: util.LoggerType = None,
) -> str:
    """Save a set of halo centers to generate maps around."""
    start = time.time()

    if type(config) is str:
        config = Config(config)

    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    box_size = config.box_sizes[sim_idx]
    save_dir = config.slice_paths[sim_idx]

    mass_dset = config.mass_dset
    mass_range = config.mass_range
    coord_dset = config.coord_dset
    extra_dsets = config.extra_dsets
    save_dir = config.coords_paths[sim_idx]
    coords_fname = config.coords_name
    sample_haloes_bins = config.sample_haloes_bins

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            config=config,
            fname=f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_save_coords{config.log_name_append}",
        )

    if sim_suite.lower() == "bahamas":
        fname = bahamas.save_coords_file(
            sim_dir=str(sim_dir),
            snapshot=snapshot,
            mass_dset=mass_dset,
            coord_dset=coord_dset,
            mass_range=mass_range,
            extra_dsets=extra_dsets,
            save_dir=save_dir,
            coords_fname=coords_fname,
            sample_haloes_bins=sample_haloes_bins,
            logger=logger,
            verbose=False,
        )

    elif sim_suite.lower() == "miratitan":
        fname = mira_titan.save_coords_file(
            sim_dir=str(sim_dir),
            box_size=box_size,
            snapshot=snapshot,
            mass_range=mass_range,
            save_dir=save_dir,
            coords_fname=coords_fname,
            sample_haloes_bins=sample_haloes_bins,
            logger=logger,
        )

    end = time.time()
    if logger:
        logger.info(
            f"save_coords_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
    # return fname


def map_coords(
    sim_idx: int,
    snapshot: int,
    slice_axis: int,
    coords_file: str,
    config: Union[Config, str],
    logger: util.LoggerType = None,
) -> List[str]:
    """Save a set of maps for sim_idx in config.sim_paths."""
    start = time.time()

    if type(config) is str:
        config = Config(config)

    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]

    coords_name = config.coords_name

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            config=config,
            fname=f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_map_coords{config.log_name_append}",
        )

    slice_dir = config.slice_paths[sim_idx]
    num_slices = config.num_slices

    save_dir = config.map_paths[sim_idx]
    map_name_append = config.map_name_append
    map_overwrite = config.map_overwrite
    map_method = config.map_method

    map_types = config.map_types[sim_idx]
    map_pix = config.map_pix
    map_size = config.map_size
    map_thickness = config.map_thickness
    n_ngb = config.n_ngb

    with h5py.File(str(coords_file), "r") as h5file:
        centers = h5file["coordinates"][:] * u.Unit(
            h5file["coordinates"].attrs["units"]
        )
        group_ids = h5file["group_ids"][:]
        masses = h5file["masses"][:] * u.Unit(h5file["masses"].attrs["units"])

    fname = map_gen.save_maps(
        centers=centers,
        group_ids=group_ids,
        masses=masses,
        slice_dir=slice_dir,
        snapshot=snapshot,
        slice_axis=slice_axis,
        num_slices=num_slices,
        box_size=box_size,
        map_pix=map_pix,
        map_size=map_size,
        map_thickness=map_thickness,
        map_types=map_types,
        save_dir=save_dir,
        coords_name=coords_name,
        map_name_append=map_name_append,
        overwrite=map_overwrite,
        method=map_method,
        n_ngb=n_ngb,
        logger=logger,
        verbose=False,
    )

    end = time.time()
    if logger:
        logger.info(
            f"map_coords_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
    # return fname


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
