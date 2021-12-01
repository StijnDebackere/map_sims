from typing import Optional

import astropy.units as u
import numpy as np


def dsigma_from_sigma(map_sigma: u.Quantity, pix_size: u.Quantity) -> u.Quantity:
    """Convert projected surface mass density map to the complex Delta Sigma map.

    Parameters
    ----------
    map_sigma : (n, n) astropy.units.Quantity
        projected surface mass density map
    pix_size : astropy.units.Quantity
        physical pixel size

    Returns
    -------
    Delta_Sigma : (n, n) astropy.units.Quantity
        complex excess surface mass density
    """
    dims = map_sigma.shape

    # zero frequency at [0, 0]
    x, y = np.meshgrid(
        np.fft.fftfreq(dims[0]) * dims[0],
        np.fft.fftfreq(dims[1]) * dims[1],
        indexing="ij",
    )
    i, j = x / (dims[0] * pix_size), y / (dims[1] * pix_size)

    kx = 2 * np.pi * i
    ky = 2 * np.pi * j
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = np.nan

    # ensure zero frequency at [0, 0]
    map_sigma_fft = np.fft.fft2(np.fft.ifftshift(map_sigma))

    # dsigma is the unscaled shear
    dsigma_1_fft = 1 / k2 * (kx ** 2 - ky ** 2) * map_sigma_fft
    dsigma_2_fft = 2 * kx * ky / k2 * map_sigma_fft

    dsigma_1_fft[0, 0] = 0.0 * dsigma_1_fft.unit
    dsigma_2_fft[0, 0] = 0.0 * dsigma_2_fft.unit

    # shift zero frequency back to [dims // 2, dims // 2]
    dsigma_1 = np.fft.fftshift(np.real(np.fft.ifft2(dsigma_1_fft)))
    dsigma_2 = np.fft.fftshift(np.real(np.fft.ifft2(dsigma_2_fft)))

    dsigma = dsigma_1 + 1j * dsigma_2
    return dsigma


def sigma_from_shear(
    map_shear: u.Quantity, pix_size: u.Quantity, sigma_crit: u.Quantity
) -> u.Quantity:
    """Convert complex shear map to projected surface mass density map.

    Parameters
    ----------
    map_shear : (n, n) astropy.units.Quantity
        shear map
    pix_size : astropy.units.Quantity
        physical pixel size

    Returns
    -------
    sigma : (n, n) astropy.units.Quantity
        projected surface mass density
    """
    dims = map_shear.shape

    # zero frequency at [0, 0]
    x, y = np.meshgrid(
        np.fft.fftfreq(dims[0]) * dims[0],
        np.fft.fftfreq(dims[1]) * dims[1],
        indexing="ij",
    )
    i, j = x / (dims[0] * pix_size), y / (dims[1] * pix_size)

    kx = 2 * np.pi * i
    ky = 2 * np.pi * j
    k2 = kx ** 2 + ky ** 2

    # ensure zero frequency at [0, 0]
    map_shear_fft = np.fft.fft2(np.fft.ifftshift(map_shear))

    # dsigma is the unscaled shear
    sigma_1_fft = 1 / k2 * (kx ** 2 - ky ** 2) * map_shear_fft
    sigma_2_fft = 2 * kx * ky / k2 * map_shear_fft
    sigma_fft = sigma_1_fft + sigma_2_fft
    sigma_fft[0, 0] = 0.0 * sigma_fft.unit

    # shift zero frequency back to [dims // 2, dims // 2]
    sigma = -np.fft.fftshift(np.real(np.fft.ifft2(sigma_fft)))

    return sigma


def shear_red_from_sigma(
    map_sigma: u.Quantity,
    pix_size: u.Quantity,
    sigma_crit: u.Quantity,
    dsigma: Optional[u.Quantity] = None,
) -> u.Quantity:
    """Convert projected surface mass density map to reduced tangential shear map.

    Parameters
    ----------
    map_sigma : (n, n) astropy.units.Quantity
        projected surface mass density map
    pix_size : astropy.units.Quantity
        physical pixel size
    sigma_crit : astropy.units.Quantity
        critical surface mass density
    dsigma : Optional[astropy.units.Quantity]
        differential surface mass density for map_sigma

    Returns
    -------
    Delta_Sigma : (n, n) astropy.units.Quantity
        tangential excess surface mass density
    """
    if disgma is None:
        dsigma = dsigma_from_sigma(map_sigma=map_sigma, pix_size=pix_size)

    g = (dsigma / sigma_crit) / (1 - map_sigma / sigma_crit)
    return g
