import logging
import os
from pathlib import Path
import time
import traceback
from typing import List, Optional, Union

import astropy.units as u
import h5py
import numpy as np

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
        level=level

    )
    # ensure that we have different loggers for each simulation and snapshot
    # in multiprocessing, PID can be the same across snapshots and sim_idx
    logger = logging.getLogger(f"{os.getpid()} - {config.sim_dirs[sim_idx]} - {snapshot:03d}")

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
    downsample = config.slice_downsample
    downsample_factor = config.downsample_factor
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_slice_sim{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    if sim_suite.lower() == "bahamas":
        try:
            fnames = bahamas.save_slice_data(
                sim_dir=str(sim_dir),
                snapshot=snapshot,
                ptypes=ptypes,
                slice_axes=slice_axes,
                num_slices=num_slices,
                downsample=downsample,
                downsample_factor=downsample_factor,
                save_dir=save_dir,
                verbose=False,
                logger=logger,
            )
        except Exception as e:
            fnames = []

            if logger:
                logger.error("Failed with exception:", exc_info=True)
                with open(
                    f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                    "w"
                ) as f:
                    f.write(traceback.format_exc())

    elif sim_suite.lower() == "miratitan":
        try:
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
            f"slice_sim_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fnames


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


def save_subvolumes(
    sim_idx: int,
    snapshot: int,
    n_divides: int,
    config: Union[Config, str],
    curve_ids: List[int] = None,
    n_sub: int = None,
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

    if curve_ids is None:
        # extract coordinate ranges for n_sub volumes
        coord_ranges, curve_ids = util.get_subvolume_ranges(
            box_size=box_size, n_divides=n_divides, n_sub=n_sub
        )
    else:
        coord_ranges, curve_ids = util.get_subvolume_ranges(
            box_size=box_size, n_divides=n_divides, curve_ids=curve_ids
        )
    coords_fnames = [
        f"{config.coords_name}_ndiv_{n_divides:d}_id_{curve_id:d}"
        for curve_id in curve_ids
    ]
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_save_subvols{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    if sim_suite.lower() == "bahamas":
        try:
            fnames = bahamas.save_subvolumes(
                sim_dir=str(sim_dir),
                snapshot=snapshot,
                mass_dset=mass_dset,
                coord_dset=coord_dset,
                coord_ranges=coord_ranges,
                mass_range=mass_range,
                extra_dsets=extra_dsets,
                save_dir=save_dir,
                coords_fnames=coords_fnames,
                sample_haloes_bins=sample_haloes_bins,
                logger=logger,
                verbose=False,
            )
        except Exception as e:
            fnames = []

            if logger:
                logger.error("Failed with exception:", exc_info=True)
                with open(
                    f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                    "w"
                ) as f:
                    f.write(traceback.format_exc())

    elif sim_suite.lower() == "miratitan":
        try:
            fnames = mira_titan.save_subvolumes(
                sim_dir=str(sim_dir),
                box_size=box_size,
                snapshot=snapshot,
                mass_range=mass_range,
                coord_ranges=coord_ranges,
                save_dir=save_dir,
                coords_fnames=coords_fnames,
                logger=logger,
            )
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
            f"save_subvols_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fnames, curve_ids


def map_subvolumes(
    sim_idx: int,
    snapshot: int,
    slice_axis: int,
    n_divides: int,
    config: Union[Config, str],
    curve_ids: List[int] = None,
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
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_map_subvols{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
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

    # simple name without snapshot and hdf5 extension to use for saved filename
    coords_names = [f"{config.coords_name}_ndiv_{n_divides:d}_id_{curve_id:d}"
        for curve_id in curve_ids
    ]
    # full filename for loading of centers
    coords_files = [
        f"{save_dir}/{coords_name}_{snapshot:03d}.hdf5" for coords_name in coords_names
    ]

    fnames = []
    for idx, coords_file in enumerate(coords_files):
        tc0 = time.time()
        try:
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
                coords_name=coords_names[idx],
                map_name_append=map_name_append,
                overwrite=map_overwrite,
                method=map_method,
                n_ngb=n_ngb,
                logger=logger,
                verbose=False,
            )
            fnames.append(fname)
        except Exception as e:
            if logger:
                logger.error("Failed with exception:", exc_info=True)
                with open(
                    f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                    "w"
                ) as f:
                    f.write(traceback.format_exc())

        tc1 = time.time()
        if logger:
            logger.debug(f"{curve_ids[idx]=} took {tc1 - tc0:.2f}s")

    end = time.time()
    if logger:
        logger.info(
            f"map_coords_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fnames


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

    swmr = config.swmr

    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]

    coords_name = config.coords_name
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_map_coords{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    slice_dir = config.slice_paths[sim_idx]
    num_slices = config.num_slices
    downsample = config.slice_downsample
    downsample_factor = config.downsample_factor

    save_dir = config.map_paths[sim_idx]
    map_name_append = config.map_name_append
    map_overwrite = config.map_overwrite
    map_method = config.map_method

    map_types = config.map_types[sim_idx]
    map_pix = config.map_pix
    map_size = config.map_size
    map_thickness = config.map_thickness
    n_ngb = config.n_ngb

    try:
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
            downsample=downsample,
            downsample_factor=downsample_factor,
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
            swmr=swmr,
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

    end = time.time()
    if logger:
        logger.info(
            f"map_coords_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
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
    config: Union[Config, str],
    logger: util.LoggerType = None,
) -> List[str]:
    """Save a set of maps for sim_idx in config.sim_paths."""
    start = time.time()

    if type(config) is str:
        config = Config(config)

    swmr = config.swmr

    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite
    box_size = config.box_sizes[sim_idx]
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_map_full{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    slice_axes = config.slice_axes
    downsample = config.slice_downsample
    downsample_factor = config.downsample_factor

    save_dir = config.map_paths[sim_idx]
    map_name_append = config.map_name_append
    map_overwrite = config.map_overwrite
    map_method = config.map_method

    map_types = config.map_types[sim_idx]
    map_pix = config.map_pix
    n_ngb = config.n_ngb

    if sim_suite.lower() == "bahamas":
        try:
            fnames = bahamas.save_full_maps(
                sim_dir=str(sim_dir),
                snapshot=snapshot,
                slice_axes=slice_axes,
                box_size=box_size,
                map_types=map_types,
                map_pix=map_pix,
                save_dir=save_dir,
                map_name_append=map_name_append,
                downsample=downsample,
                downsample_factor=downsample_factor,
                overwrite=map_overwrite,
                swmr=swmr,
                method=map_method,
                n_ngb=n_ngb,
                logger=logger,
                verbose=False,
            )
        except Exception as e:
            fnames = []

            if logger:
                logger.error("Failed with exception:", exc_info=True)
                with open(
                    f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.err",
                    "w"
                ) as f:
                    f.write(traceback.format_exc())

    elif sim_suite.lower() == "miratitan":
        try:
            fnames = mira_titan.save_full_maps(
                sim_dir=str(sim_dir),
                snapshot=snapshot,
                slice_axes=slice_axes,
                box_size=box_size,
                map_pix=map_pix,
                save_dir=save_dir,
                map_name_append=map_name_append,
                downsample=downsample,
                downsample_factor=downsample_factor,
                overwrite=map_overwrite,
                swmr=swmr,
                method=map_method,
                n_ngb=n_ngb,
                logger=logger,
                verbose=False,
            )
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
            f"map_full_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fnames


def map_los(
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

    swmr = config.swmr

    base_dir = config.base_dir
    sim_dir = config.sim_paths[sim_idx]
    sim_suite = config.sim_suite

    box_size = config.box_sizes[sim_idx]
    ptypes = config.ptypes[sim_idx]
    log_fname = f"{config.sim_dirs[sim_idx]}_{snapshot:03d}_map_los{config.log_name_append}"

    if logger is None and config.logging:
        logger = get_logger(
            sim_idx=sim_idx,
            snapshot=snapshot,
            config=config,
            fname=log_fname,
        )

    slice_dir = config.slice_paths[sim_idx]
    num_slices = config.num_slices
    downsample = config.slice_downsample
    downsample_factor = config.downsample_factor

    save_dir = config.map_paths[sim_idx]
    map_name_append = config.map_name_append
    map_overwrite = config.map_overwrite
    map_method = config.map_method

    map_types = config.map_types[sim_idx]
    map_size = config.map_size
    map_thickness = config.map_thickness
    map_pix = config.map_pix
    n_ngb = config.n_ngb

    coords_name = config.coords_name
    with h5py.File(str(coords_file), "r") as h5file:
        centers = h5file["coordinates"][:] * u.Unit(
            h5file["coordinates"].attrs["units"]
        )
        group_ids = h5file["group_ids"][:]
        masses = h5file["masses"][:] * u.Unit(h5file["masses"].attrs["units"])


    if sim_suite.lower() == "bahamas":
        try:
            fname = bahamas.save_maps_los(
                sim_dir=sim_dir,
                snapshot=snapshot,
                centers=centers,
                group_ids=group_ids,
                masses=masses,
                slice_dir=slice_dir,
                slice_axis=slice_axis,
                num_slices=num_slices,
                downsample=downsample,
                downsample_factor=downsample_factor,
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
                swmr=swmr,
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
            fname = mira_titan.save_maps_los(
                sim_dir=sim_dir,
                snapshot=snapshot,
                slice_axis=slice_axis,
                box_size=box_size,
                map_pix=map_pix,
                map_thickness=map_thickness,
                save_dir=save_dir,
                map_name_append=map_name_append,
                downsample=downsample,
                downsample_factor=downsample_factor,
                overwrite=map_overwrite,
                swmr=swmr,
                method=map_method,
                n_ngb=n_ngb,
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

    end = time.time()
    if logger:
        logger.info(
            f"map_los_{config.sim_dirs[sim_idx]}_{snapshot:03d} took {end - start:.2f}s"
        )
        with open(
            f"{config.log_dir}/{log_fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.complete",
            "w"
        ) as f:
            pass

    return fname
