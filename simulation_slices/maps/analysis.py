from typing import Callable, Optional, Union

import astropy.units as u
import h5py
import numpy as np
import pandas as pd

from simulation_slices import Config
import simulation_slices.maps.generation as map_gen
import simulation_slices.maps.observables as obs
import simulation_slices.maps.tools as tools
import simulation_slices.utilities as util


PI = np.pi
PI_INV = 1.0 / np.pi


def filter_zeta(R, maps, A_pix, R1, R2, Rm, **kwargs):
    """Convergence map filter for zeta_c statistic of Clowe et al.
    (1998).

    Parameters
    ----------
    R : (map_pix**2,) astropy.units.Quantity
        radial distance from center for each pixel
    maps : (..., map_pix, map_pix) array
        map values at each R
    A_pix : astropy.units.Quantity
        pixel area
    R1 : astropy.units.Quantity
        inner radius
    R2 : astropy.units.Quantity
        inner annulus radius
    Rm : astropy.units.Quantity
        outer annulus radius

    Returns
    -------
    M_ap: astropy.units.Quantity
        aperture mass
    """
    R = np.atleast_1d(R)
    if len(R.shape) > 1:
        raise ValueError("R should be pix_id array with shape (map_pix**2)")

    maps_R1 = (
        np.sum(PI_INV / R1 ** 2 * maps.reshape(-1, R.shape[0])[..., R <= R1], axis=-1)
        * A_pix
    )
    maps_R2_Rm = (
        np.sum(
            -PI_INV
            / (Rm ** 2 - R2 ** 2)
            * maps.reshape(-1, R.shape[0])[..., ((R > R2) & (R < Rm))],
            axis=-1,
        )
        * A_pix
    )
    return PI * R1 ** 2 * (maps_R1 + maps_R2_Rm)


def M_aperture(
    maps: u.Quantity,
    pix_size: u.Quantity,
    filt: Callable,
    r_off: u.Quantity,
    **filt_kwargs,
) -> u.Quantity:
    """Filter maps around x_0 with filter filt(x - x_0, **filt_kwargs).

    Parameters
    ----------
    maps : (..., n_pix, n_pix) astropy.units.Quantity
        projected mass maps with same units as sigma_crit
    filt : callable
        filter function (R_pix_id, Sigma, A_pix, **filt_kwargs)
    r_off : (2, ) astropy.units.Quantity
        offset with respect to map center to center on
    pix_size : astropy.units.Quantity
        physical scale of pixels in length units matching sigma_crit

    Returns
    -------
    M_ap : astropy.units.Quantity
        mass contained within filt around x_0
    """
    map_pix = maps.shape[-1]
    center = (0.5 * map_pix, 0.5 * map_pix)
    r_off = np.atleast_1d(r_off)

    R = (
        tools.pix_dist(
            tools.pix_id_to_pixel(np.arange(map_pix ** 2), map_pix),
            center + r_off / pix_size,
            b_is_pix=False,
        )
        * pix_size
    )
    A_pix = pix_size ** 2
    M_ap = filt(R, maps, A_pix, **filt_kwargs)

    return M_ap


def radial_average(
    maps: u.Quantity,
    r_bins: u.Quantity,
    pix_size: u.Quantity,
    r_off: Optional[np.ndarray] = None,
    log: bool = False,
) -> u.Quantity:
    """Average maps in r_bins around r_off from map center with pix_size
    for pixel length scale."""
    map_pix = maps.shape[-1]
    # get map center
    pix_center = np.array(maps.shape[-2:]) / 2

    if r_off is None:
        r_off = np.array([0, 0]) * pix_size

    # get distance from pixels to center
    pix_range = np.arange(0, map_pix, 1)
    pix_dists = (
        np.linalg.norm(
            util.arrays_to_coords(pix_range, pix_range)
            - (pix_center + r_off / pix_size),
            axis=-1,
        )
        * pix_size
    )

    radial_profile = util.apply_grouped(
        fun=np.median,
        grouped_data=util.groupby(
            data=maps.flatten(),
            index=pix_dists,
            bins=r_bins,
        ),
        axis=-1,
    )
    r = util.bin_centers(r_bins, log=log)

    return (r, np.array([float(r.value) for r in radial_profile.values()]) * r.unit)



    """
    pass
