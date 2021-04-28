import numpy as np

import simulation_slices.maps.tools as tools
import simulation_slices.utilities as util


PI = np.pi
PI_INV = 1. / np.pi


def filter_zeta(R, maps, A_pix, R1, R2, Rm, **kwargs):
    """Convergence map filter for zeta_c statistic of Clowe et al.
    (1998)."""
    maps_R1 = np.sum(
        PI_INV / R1**2 * maps[..., np.where(R <= R1)],
        axis=(-2, -1)
    ) * A_pix
    maps_R2_Rm = np.sum(
        -PI_INV / (Rm**2 - R2**2) * maps[..., np.where((R > R2) & (R < Rm))],
        axis=(-2, -1)
    ) * A_pix

    return PI * R1**2 * (maps_R1 + maps_R2_Rm)


def M_aperture(
        maps, sigma_crit, filt, pix_scale, pix_0, **filt_kwargs):
    """Filter maps around x_0 with filter filt(pix - pix_0, **filt_kwargs).

    Parameters
    ----------
    maps : (..., n_pix, n_pix) array
        projected mass maps with same units as sigma_crit
    sigma_crit : float
        critical surface mass density
    filt : callable
        filter function (R, Sigma, A_pix, **filt_kwargs)
    pix_scale : float
        physical scale of pixels in length units matching sigma_crit
    x_0 : (2,) array
        pixel to center filter on

    Returns
    -------
    M_ap : float
        mass contained within filt around x_0

    """
    num_pix_side = maps.shape[-1]
    pixels = tools.pix_id_to_pixel(
        np.arange(num_pix_side**2), num_pix_side)

    R_pix = np.linalg.norm(pixels - pix_0.reshape(2, 1), axis=0)
    R = (R_pix * pix_scale).reshape(num_pix_side, num_pix_side)
    A_pix = pix_scale**2
    M_ap = filt(R, maps, A_pix, **filt_kwargs)
    return M_ap


def Y_sz(
        maps, filt, pix_scale, pix_0, **filt_kwargs):
    """Filter maps around x_0 with filter filt(pix - pix_0, **filt_kwargs).

    Parameters
    ----------
    maps : (..., n_pix, n_pix) array
        projected mass maps with same units as sigma_crit
    sigma_crit : float
        critical surface mass density
    filt : callable
        filter function (R, Sigma, A_pix, **filt_kwargs)
    pix_scale : float
        physical scale of pixels in length units matching sigma_crit
    x_0 : (2,) array
        pixel to center filter on

    Returns
    -------
    M_ap : float
        mass contained within filt around x_0

    """
    pass
