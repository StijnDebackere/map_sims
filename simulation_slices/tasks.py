import logging
import os
from pathlib import Path
import time
import traceback
from typing import List, Optional, Union

import astropy.units as u
import h5py
import numpy as np
from numpy.random import Generator

from simulation_slices import Config
import simulation_slices.utilities as util
import simulation_slices.maps.generation as map_gen
import simulation_slices.sims.bahamas as bahamas
import simulation_slices.sims.mira_titan as mira_titan


def get_logger(sim_idx: int, snapshot: int, config: Config, fname: str) -> logging.Logger:
    log_dir = config.log_dir
    log_fname = f"{log_dir}/{fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.log"

    if config.log_level.lower() == "info":
        level = logging.INFO
    elif config.log_level.lower() == "debug":
        level = logging.DEBUG
    elif config.log_level.lower() == "warning":
        level = logging.WARNING
    elif config.log_level.lower() == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_fname,
        filemode="w",
        format="%(asctime)s - %(name)s [%(levelname)s] %(funcName)s - %(message)s",
        level=level,
        force=True
    )
    # ensure that we have different loggers for each simulation and snapshot
    # in multiprocessing, PID can be the same across snapshots and sim_idx
    logger = logging.getLogger(f"{os.getpid()} - {config.sim_dirs[sim_idx]} - {snapshot:03d}")

    return logger


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

    mass_dset = config.mass_dset
    mass_range = config.mass_range
    coord_dset = config.coord_dset
    extra_dsets = config.extra_dsets
    save_dir = config.coords_paths[sim_idx]
    coords_fname = config.coords_name
    sample_haloes_bins = config.sample_haloes_bins
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_save_coords{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    if sim_suite.lower() == "bahamas":
        try:
            fname = bahamas.save_halo_coords_file(
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
        except Exception as e:
            fname = []

            if logger:
                logger.error("Failed with exception:", exc_info=True)
                with open(
                    f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                    "w"
                ) as f:
                    f.write(traceback.format_exc())

    elif sim_suite.lower() == "miratitan":
        try:
            fname = mira_titan.save_halo_coords_file(
                sim_dir=str(sim_dir),
                snapshot=snapshot,
                mass_range=mass_range,
                save_dir=save_dir,
                coords_fname=coords_fname,
                sample_haloes_bins=sample_haloes_bins,
                logger=logger,
            )
        except Exception as e:
            fname = []

            if logger:
                logger.error("Failed with exception:", exc_info=True)
                with open(
                    f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                    "w"
                ) as f:
                    f.write(traceback.format_exc())

    end = time.time()
    if logger:
        logger.info(
            f"save_coords_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fname


def map_full(
    sim_idx: int,
    snapshot: int,
    slice_axis: int,
    config: Union[Config, str],
    logger: util.LoggerType = None,
    rng: Optional[Generator] = None,
) -> List[str]:
    """Save a set of maps for sim_idx in config.sim_paths."""
    start = time.time()

    if type(config) is str:
        config = Config(config)

    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    box_size = config.box_sizes[sim_idx]
    log_fname = str(
        f"{config.sim_dirs[sim_idx]}_slice_{slice_axis}_"
        f"snap_{snapshot:03d}_map_full{config.log_name_append}"
    )

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    iterate_files = config.iterate_files
    scramble_files = config.scramble_files

    save_dir = config.map_paths[sim_idx]
    map_name_append = config.map_name_append
    map_overwrite = config.map_overwrite
    map_method = config.map_method

    map_thickness = config.map_thickness
    map_types = config.map_types[sim_idx]
    map_pix = config.map_pix
    n_ngb = config.n_ngb

    fnames = []
    try:
        fname = map_gen.save_map_full(
            sim_suite=sim_suite.lower(),
            sim_dir=str(sim_dir),
            snapshot=snapshot,
            slice_axis=slice_axis,
            box_size=box_size,
            map_pix=map_pix,
            map_thickness=map_thickness,
            map_types=map_types,
            save_dir=save_dir,
            map_name_append=map_name_append,
            overwrite=map_overwrite,
            iterate_files=iterate_files,
            scramble_files=scramble_files,
            method=map_method,
            n_ngb=n_ngb,
            logger=logger,
            rng=rng,
            verbose=False,
        )
        fnames.append(fname)
    except Exception as e:
        fnames = []

        if logger:
            logger.error("Failed with exception:", exc_info=True)
            with open(
                f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                "w"
            ) as f:
                f.write(traceback.format_exc())

    end = time.time()
    if logger:
        logger.info(
            f"map_full_{slice_axis}_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fnames
