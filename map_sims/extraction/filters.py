from typing import Union, Optional, Tuple

import astropy.units as u
import numpy as np


PI = np.pi
PI_INV = 1.0 / np.pi


def sigma_mean(
    R: u.Quantity,
    sigma: u.Quantity,
    A_pix: u.Quantity,
    R0: u.Quantity,
    R1: u.Quantity,
    **kwargs,
) -> u.Quantity:
    """Calculate the mean projected mass between R0 and R1 from surface
    mass density sigma.

    Parameters
    ----------
    R : (map_pix, map_pix) astropy.units.Quantity
        radial distance from center for each pixel
    sigma : (..., map_pix, map_pix) astropy.units.Quantity
        projected surface mass density at each R
    A_pix : length ** 2, astropy.units.Quantity
        pixel area
    R0 : length, astropy.units.Quantity
        inner radius
    R1 : length, astropy.units.Quantity
        outer radius

    Returns
    -------
    M : astropy.units.Quantity
        aperture mass
    """
    R = np.atleast_2d(R)
    if len(R.shape) != 2 and R.shape != sigma.shape[-2:]:
        raise ValueError(f"R needs to match final 2 dimensions of sigma: {sigma.shape}")

    selection = (R >= R0) & (R <= R1)
    norm = PI_INV * A_pix / (R1 ** 2 - R0 ** 2)
    sigma_mean = norm * np.sum(sigma[..., selection], axis=-1)

    return sigma_mean


def filter_zeta(
    R: u.Quantity,
    maps: u.Quantity,
    A_pix: u.Quantity,
    R1: u.Quantity,
    R2: Optional[u.Quantity] = None,
    Rm: Optional[u.Quantity] = None,
    return_bg: bool = False,
    **kwargs,
) -> Union[u.Quantity, Tuple[u.Quantity, u.Quantity]]:
    """Convergence map filter for zeta_c statistic of Clowe et al.
    (1998).

    Parameters
    ----------
    R : (map_pix, map_pix) astropy.units.Quantity
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
    return_bg : bool
        return background mass from annulus estimation

    Returns
    -------
    M_ap : astropy.units.Quantity
        aperture mass
    if return_bg:
        M_bg : astropy.units.Quantity
            background mass
    """
    R = np.atleast_2d(R)
    if len(R.shape) != 2 and R.shape != maps.shape[-2:]:
        raise ValueError(f"R needs to match final 2 dimensions of maps: {maps.shape}")

    sigma_R1 = sigma_mean(R=R, sigma=maps, A_pix=A_pix, R0=0 * R1.unit, R1=R1)
    M_R1 = PI * R1 ** 2 * sigma_R1
    M_bg = None

    if R2 is not None and Rm is not None:
        sigma_bg = sigma_mean(R=R, sigma=maps, A_pix=A_pix, R0=R2, R1=Rm)
        M_bg = PI * R1 ** 2 * sigma_bg
        M_zeta = M_R1 - M_bg
    else:
        M_zeta = M_R1

    result = M_zeta
    if return_bg and M_bg is not None:
        result = (M_zeta, M_bg)

    return result
