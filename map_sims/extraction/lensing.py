import logging
import os
import time
from typing import List, Optional, Tuple, Union, Any

import astropy.units as u
import numpy as np
from tqdm import tqdm

import map_sims.extraction.data as data
import map_sims.extraction.filters as filters
import map_sims.io as io
import map_sims.lensing.generate_lensing as gen_lensing
import map_sims.maps.operations as map_ops


def compute_lensing_maps(
    map_full: u.Quantity,
    pix_size: u.Quantity,
    box_size: u.Quantity,
    map_thickness: u.Quantity,
    cut_map_size: u.Quantity,
    coords: u.Quantity,
    verbose: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """Extract maps of cut_map_size around coords in map_full.

    Parameters
    ----------
    map_full : astropy.units.Quantity
        map to compute masses from
    pix_size : astropy.units.Quantity
        size of map pixels
    box_size : astropy.units.Quantity
        size of simulation box
    map_thickness : astropy.units.Quantity
        thickness of the map
    cut_map_size : astropy.units.Quantity
        size of the lensing map region
    coords : astropy.units.Quantity
        coordinates of map_full to center on
    verbose : bool
        print outputs
    logger : logging.Logger
        log info to logger

    Returns
    -------
    results : dict
        aperture masses and metadata for calculation

    """
    A_pix = pix_size ** 2

    results = {
        "pix_size": pix_size,
        "box_size": box_size,
        "map_thickness": map_thickness,
        "cut_map_size": cut_map_size,
        "maps_sigma": [],
        "maps_dsigma": [],
    }

    iterator = enumerate(coords)
    if verbose:
        iterator = tqdm(iterator, total=coords.shape[0], desc="Computing lensing maps")

    for idx, center in iterator:
        # get the pixel grid and corresponding distances for the cut-out around center
        map_cutout, dists = map_ops.slice_map_around_center(
            center=center,
            map_full=map_full,
            map_cutout_size=cut_map_size,
            pix_size=pix_size,
            box_size=box_size,
        )
        dsigma = gen_lensing.dsigma_from_sigma(
            map_sigma=map_cutout,
            pix_size=pix_size,
        )
        results["maps_sigma"].append(map_cutout)
        results["maps_dsigma"].append(dsigma)

        if idx % 10000 == 0 and logger:
            logger.info(
                f"{idx}/{coords.shape[0]}  <=> {idx / coords.shape[0] * 100:.0f}%"
            )

    results["maps_sigma"] = np.stack(results["maps_sigma"], axis=0)
    results["maps_dsigma"] = np.stack(results["maps_dsigma"], axis=0)
    return results


def save_lensing_maps(
    fname: str,
    sim: str,
    slice_axis: int,
    snapshot: int,
    info_file: str,
    extra_dsets: dict,
    map_file: str,
    cut_map_size: u.Quantity,
    overwrite: bool = False,
    verbose: bool = False,
    sim_suite: str = "bahamas",
    logger: logging.Logger = None,
    selection: np.ndarray = None,
) -> None:
    """Save sigma and dsigma maps for coords in info_file and map in map_file.

    Parameters
    ----------
    fname : str
        path to save lensing maps to
    sim : str
        simulation name
    slice_axis : int
        axis along which map is projected [0: x, 1: y, 2: z]
    snapshot : int
        snapshot of the map
    info_file : str
        filename with halo centers and metadata
    extra_dsets : dict
        mapping between names to save and dset names in info_file
    map_file : str
        filename for the map to calculate aperture masses from
    cut_map_size : astropy.units.Quantity
        size of cutouts to slice around halo centers > 2 * r_aps.max()
    overwrite : bool
        overwrite fname
    verbose : bool
        verbose outputs
    sim_suite : str
        simulation suite ["bahamas", "miratitan"]

    Returns
    -------

    """
    t0 = time.time()
    if logger:
        logger.info(f"Loading results from {info_file}")
    # start loading information for all groups for sim
    sim_results = data.load_from_info_files(
        sims=[sim],
        info_files=[info_file],
        extra_dsets=extra_dsets,
        selection=selection,
    )

    map_full, metadata = data.load_map_file(sim, map_file, sim_suite=sim_suite, logger=logger)
    no_slice_axis = np.arange(0, 3) != slice_axis

    if logger:
        map_file_split = map_file.split("/")
        sl = int(map_file_split[-1][0])
        if info_file.endswith(".hdf5"):
            snap = int(info_file[:-5][-3:])
        else:
            raise ValueError(f"{info_file=} should have .hdf5 extension.")

        logger = logger.getChild(f"{os.getpid()}-{sim}_{snap:03d}_{sl}")
        logger.info(f"Starting to compute aperture_masses for {info_file=}")

    maps = compute_lensing_maps(
        map_full=map_full,
        pix_size=metadata["pix_size"],
        box_size=metadata["box_size"],
        map_thickness=metadata["map_thickness"],
        coords=sim_results[sim]["coordinates"][:, no_slice_axis],
        cut_map_size=cut_map_size,
        verbose=verbose,
        logger=logger,
    )

    results = {}
    # only called for single snapshot and slice_axis, unpack to results
    results[sim] = {
        f"snap_{int(snapshot):03d}": {
            **sim_results[sim],
            "info_file": info_file,
            "z": metadata["z"],
            f"slice_{int(slice_axis):d}": {
                "map_file": map_file,
                **maps,
            },
        },
    }
    io.dict_to_hdf5(
        fname=fname,
        data=results,
        overwrite=overwrite,
    )
    t1 = time.time()
    if logger:
        logger.info(f"Saved results to {fname=}, finished in {t1 - t0:.2f}s")
