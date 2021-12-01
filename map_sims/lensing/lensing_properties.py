import astropy.constants as constants
from astropy.cosmology import FlatwCDM
import astropy.units as u
import numpy as np

import lensing_sims.tools as tools


def sigma_crit(
    z_l,
    beta_mean,
    cosmo={"omega_m": 0.315, "w0": -1.0, "h": 0.7},
    littleh=True,
    comoving=True,
):
    """Return the critical surface mass density for a lens as z_l with
    mean lensing efficiency beta_mean.

    Parameters
    ----------
    z_l : array-like
        redshifts
    beta_mean : float
        mean lensing efficiency
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0
    littleh : bool
        return littleh units
    comoving : bool
        variables are in comoving coordinates

    Returns
    -------
    sigma_crit : array-like
        critical surface mass density for each lens
    """
    c = FlatwCDM(Om0=cosmo["omega_m"], H0=100 * cosmo["h"], w0=cosmo["w0"])
    # [(M_odot / h) / (Mpc / h)]
    alpha = (constants.c ** 2 / (4 * np.pi * constants.G)).to(u.solMass / u.Mpc)
    sigma_crit = (
        alpha * 1 / (beta_mean * c.angular_diameter_distance(z_l).to(u.Mpc))
    ).to(u.Msun / u.Mpc ** 2)

    if littleh:
        sigma_crit = sigma_crit / c.h * u.littleh

    # surface area grows in comoving units => surface mass density goes down
    if comoving:
        sigma_crit *= 1 / (1 + z_l) ** 2

    return sigma_crit


def n_mpch2(
    n_arcmin2,
    z_l,
    cosmo={"omega_m": 0.315, "w0": -1.0, "h": 0.7},
    littleh=True,
    comoving=True,
):
    """
    Convert a mean background galaxy density per arcmin^2 for a lens
    at redshift z_l to a density per (Mpc/h)^2 assuming cosmo

    Parameters
    ----------
    n_arcmin2 : float
        background galaxy density per arcmin^2
    z_l : array-like
        redshift of the lens
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0
    littleh : bool
        return littleh units
    comoving : bool
        variables are in comoving coordinates

    Returns
    -------
    n_mpch2 : array-like
        background galaxy density per (Mpc/h)^2 for each z_l
    """
    n_arcmin2 *= 1 / u.arcmin ** 2
    c = FlatwCDM(Om0=cosmo["omega_m"], H0=100 * cosmo["h"], w0=cosmo["w0"])
    # arcminute to Mpc/h conversion factor
    mpch_per_arcmin = c.kpc_proper_per_arcmin(z=z_l).to(u.Mpc / u.arcmin)
    if littleh:
        mpch_per_arcmin *= c.h / u.littleh

    # galaxy density is arcmin^-2
    nmpch2 = n_arcmin2 * mpch_per_arcmin ** (-2)
    # number density in physical units => area grows in comoving => number density down
    if comoving:
        nmpch2 *= 1 / (1 + z_l) ** 2

    return nmpch2


def shape_noise(
    R_bins,
    z_l,
    cosmo={"omega_m": 0.315, "w0": -1.0, "h": 0.7},
    sigma_e=0.25,
    n_arcmin2=10,
    littleh=True,
    comoving=True,
    log=False,
):
    """
    Calculate the uncertainty due to intrinsic shape noise for bin r_i
    due to the number of background galaxies

    Parameters
    ----------
    R_bins : (r,) array
        bin edges
    z_l : (z,) array
        redshift of the lens
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0
    sigma_e : float
        ellipticity noise per galaxy
    n_arcmin2 : float
        background galaxy density [arcmin^-2]
    littleh : bool
        return littleh units
    comoving : bool
        variables are in comoving coordinates
    log : bool
        logarithmic bins

    Returns
    -------
    shape_noise : (z, r) array
        average shape noise for each R_bin
    """
    # (1, R) array or (z, R) array
    R_bins = np.atleast_2d(R_bins)
    R_centers = tools.bin_centers(R_bins, log=log)
    # (z, 1) array
    nmpch2 = n_mpch2(
        n_arcmin2=n_arcmin2,
        z_l=z_l,
        cosmo=cosmo,
        comoving=comoving,
        littleh=littleh,
    ).reshape(-1, 1)

    # (z, R) array
    N_bins = 2 * np.pi * nmpch2 * (np.diff(R_bins) * R_centers)
    sigma_bins = sigma_e / (N_bins) ** 0.5
    return sigma_bins
