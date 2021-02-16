import numpy as np

import simulation_slices.map_tools as map_tools
import simulation_slices.utilities as util

PI = np.pi
PI_INV = 1. / np.pi


def filter_zeta(R, sigma, A_pix, R1, R2, Rm, **kwargs):
    """Convergence map filter for zeta_c statistic of Clowe et al.
    (1998)."""
    sigma_R1 = np.sum(
        PI_INV / R1**2 * sigma[np.where(R <= R1)]
    ) * A_pix
    sigma_R2_Rm = np.sum(
        -PI_INV / (Rm**2 - R2**2) * sigma[np.where((R > R2) & (R < Rm))]
    ) * A_pix

    return PI * R1**2 * (sigma_R1 + sigma_R2_Rm)


def M_ap_from_sigma_map(
        sigma_map, sigma_crit, filt, pix_scale, pix_0, **filt_kwargs):
    """Filter sigma_map around x_0 with filter filt(pix - pix_0).

    Parameters
    ----------
    sigma_map : (n_pix, n_pix) array
        projected mass map with same units as sigma_crit
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
    num_pix_side = sigma_map.shape[0]
    pixels = map_tools.pix_id_to_pixel(
        np.arange(num_pix_side**2), num_pix_side)

    R_pix = np.linalg.norm(pixels - pix_0.reshape(2, 1), axis=0)
    R = (R_pix * pix_scale).reshape(num_pix_side, num_pix_side)
    A_pix = pix_scale**2
    M_ap = filt(R, sigma_map, A_pix, **filt_kwargs)
    return M_ap


def analyze_maps(maps, func, **func_kwargs):
    """Analyze the given maps with func that requires func_kwargs.

    Parameters
    ----------
    maps : (N, n_pix, n_pix) array
        maps to be analyzed
    func : callable
        analysis function (map, **func_kwargs)

    Returns
    -------
    results : (N, ...) array with results
    """
    results = []
    for mp in maps:
        results.append(func(mp, **func_kwargs))
    return results


def get_M_aps(maps, sigma_crit, pix_scale, R1, R2, Rm):
    """Get M_ap for all maps."""
    results = analyze_maps(
        maps=maps, func=M_ap_from_sigma_map,
        sigma_crit=1, filt=filter_zeta, pix_scale=pix_scale,
        pix_0=np.array([maps.shape[-1] // 2, maps.shape[-1] // 2]),
        R1=R1, R2=R2, Rm=Rm
    )
    return np.asarray(results)
