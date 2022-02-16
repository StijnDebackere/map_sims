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
import map_sims.maps.operations as map_ops


def compute_aperture_masses(
    map_full: u.Quantity,
    pix_size: u.Quantity,
    box_size: u.Quantity,
    map_thickness: u.Quantity,
    cut_map_size: u.Quantity,
    coords: u.Quantity,
    r_aps: u.Quantity,
    r2s: u.Quantity,
    rms: u.Quantity,
    r_sods: Optional[u.Quantity] = None,
    rho_mean: Optional[u.Quantity] = None,
    calc_bg: bool = False,
    verbose: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """Compute m_ap in r_aps around coords in map_full for sim.

    Parameters
    ----------
    map_full : astropy.units.Quantity
        map(s) to compute masses from
    pix_size : astropy.units.Quantity
        size of map pixels
    box_size : astropy.units.Quantity
        size of simulation box
    map_thickness : astropy.units.Quantity
        thickness(es) of the map
    coords : astropy.units.Quantity
        coordinates of map_full to center on
    r_aps : astropy.units.Quantity
        aperture sizes
    r2s : astropy.units.Quantity
        inner radius of background annulus for each r_ap
    rms : astropy.units.Quantity
        outer radius of background annulus for each r_ap
    rho_mean : astropy.units.Quantity
        mean background matter density of the Universe
    calc_bg : bool
        return mean background masses for each r_ap calculated from R2-Rm annulus
    r_sods: [Optional: astropy.units.Quantity]
        spherical overdensity radii
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

    r_aps = np.atleast_1d(r_aps)
    map_thickness = np.atleast_1d(map_thickness)

    results = {
        "R_aps": r_aps,
        "R2": r2s,
        "Rm": rms,
        "pix_size": pix_size,
        "box_size": box_size,
        "map_thickness": map_thickness,
        "cut_map_size": cut_map_size,
    }

    if r2s is not None and rms is not None:
        r2s = np.atleast_1d(r2s)
        rms = np.atleast_1d(rms)
        if r2s.shape != rms.shape != r_aps.shape:
            raise ValueError("r_aps, r2s and rms should have matching shapes")
    else:
        r2s = [None] * r_aps.shape[0]
        rms = [None] * r_aps.shape[0]

    if map_thickness.shape[0] > 1:
        m_shape = (coords.shape[0], map_thickness.shape[0])
    else:
        m_shape = (coords.shape[0])

    r_ap_names = data.get_r_ap_names(r_aps=r_aps, r2s=r2s, rms=rms, bg=False)
    for name in r_ap_names:
        results[name] = (
            np.zeros(m_shape, dtype=float) * map_full.unit * A_pix.unit
        )
        if r_sods is not None:
            results["m_ap_sod" + "_".join(name.split("_")[3:])] = (
                np.zeros(m_shape, dtype=float) * map_full.unit * A_pix.unit
            )

    if calc_bg:
        r_ap_names_bg = data.get_r_ap_names(r_aps=r_aps, r2s=r2s, rms=rms, bg=True)
        return_bg = True
        # save the mean background mass from matter density
        if rho_mean is not None:
            m_mean = rho_mean * map_thickness * np.pi * r_aps ** 2
            results["m_mean"] = m_mean

        for name_bg in r_ap_names_bg:
            results[name_bg] = (
                np.zeros(m_shape, dtype=float) * map_full.unit * A_pix.unit
            )
            if r_sods is not None:
                results["m_ap_sod" + "_".join(name_bg.split("_")[3:])] = (
                    np.zeros(m_shape, dtype=float) * map_full.unit * A_pix.unit
                )

    else:
        return_bg = False


    iterator = enumerate(coords)
    if verbose:
        iterator = tqdm(iterator, total=coords.shape[0], desc="Computing m_aps")

    for idx, center in iterator:
        # get the pixel grid and corresponding distances for the cut-out around center
        map_cutout, dists = map_ops.slice_map_around_center(
            center=center,
            map_full=map_full,
            map_cutout_size=cut_map_size,
            pix_size=pix_size,
            box_size=box_size,
        )

        for idx_r, (r_ap, r2, rm) in enumerate(zip(r_aps, r2s, rms)):
            name = r_ap_names[idx_r]
            res = filters.filter_u_zeta(
                R=dists,
                maps=map_cutout,
                A_pix=A_pix,
                R1=r_ap,
                R2=r2,
                Rm=rm,
                return_bg=return_bg,
            )
            if return_bg:
                results[name][idx] = res[0]
                results[f"{name}_bg"][idx] = res[1]
            else:
                results[name][idx] = res

            if r_sods is not None:
                res = filters.filter_u_zeta(
                    R=dists,
                    maps=map_cutout,
                    A_pix=A_pix,
                    R1=r_sods[idx],
                    R2=r2,
                    Rm=rm,
                    return_bg=return_bg,
                )
                if return_bg:
                    results["m_ap_sod" + "_".join(name.split("_")[3:])][idx] = res[0]
                    results["m_ap_sod_bg" + "_".join(name.split("_")[3:])][idx] = res[1]
                else:
                    results["m_ap_sod" + "_".join(name.split("_")[3:])][idx] = res

        if idx % 10000 == 0 and logger:
            logger.info(
                f"{idx}/{coords.shape[0]}  <=> {idx / coords.shape[0] * 100:.0f}%"
            )

    return results


def save_aperture_masses(
    fname: str,
    sim: str,
    info_file: str,
    extra_dsets: dict,
    map_file: str,
    cut_map_size: u.Quantity,
    r_aps: u.Quantity,
    r2s: u.Quantity,
    rms: u.Quantity,
    overwrite: bool = False,
    verbose: bool = False,
    sim_suite: str = "bahamas",
    rho_mean: u.Quantity = None,
    calc_bg: bool = True,
    logger: logging.Logger = None,
    selection: np.ndarray = None,
) -> None:
    """Save m_ap(<r_aps) for sim with full simulation maps in map_file.

    Parameters
    ----------
    fname : str
        path to save aperture masses to
    sim : str
        simulation name
    info_file : str
        filename with halo centers and metadata
    extra_dsets : dict
        mapping between names to save and dset names in info_file
    map_file : str
        filename for the map to calculate aperture masses from
    cut_map_size : astropy.units.Quantity
        size of cutouts to slice around halo centers > 2 * r_aps.max()
    r_aps : astropy.units.Quantity
        aperture sizes to compute masses for
    r2s : astropy.units.Quantity
        outer annulus inner radius
    rms : astropy.units.Quantity
        outer annulus outer radius
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
    if r2s is not None and rms is not None:
        if r2s.shape != rms.shape != r_aps.shape:
            raise ValueError("r_aps, r2s and rms should have matching shapes")

    if cut_map_size < 2 * r_aps.max():
        raise ValueError(
            f"cannot calculate r_aps={r_aps[cut_map_size < 2 * r_aps]} with {cut_map_size=}"
        )
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
    snapshot = metadata["snapshot"]
    slice_axis = metadata["slice_axis"]
    no_slice_axis = np.arange(0, 3) != slice_axis

    if logger:
        logger = logger.getChild(f"{os.getpid()}-{sim}_{snapshot:03d}_{slice_axis}")
        logger.info(f"Starting to compute aperture_masses for {info_file=}")

    masses_info = compute_aperture_masses(
        map_full=map_full,
        pix_size=metadata["pix_size"],
        box_size=metadata["box_size"],
        map_thickness=metadata["map_thickness"],
        coords=sim_results[sim]["coordinates"][:, no_slice_axis],
        r_sods=sim_results[sim]["radii"],
        cut_map_size=cut_map_size,
        r_aps=r_aps,
        r2s=r2s,
        rms=rms,
        rho_mean=rho_mean,
        calc_bg=calc_bg,
        verbose=verbose,
        logger=logger,
    )

    results = {}
    # only called for single snapshot and slice_axis, unpack to results
    results[sim] = {
        f"snap_{int(snapshot):03d}": {
            **sim_results[sim],
            "rho_mean": rho_mean,
            "info_file": info_file,
            "z": metadata["z"],
            f"slice_{int(slice_axis):d}": {
                "map_file": map_file,
                **masses_info,
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
