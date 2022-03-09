from typing import Union

import astropy.constants as constants
from astropy.cosmology import FlatwCDM, FLRW
from astropy.cosmology.units import with_H0
import astropy.units as u
import numpy as np

import map_sims.tools as tools


def convert_cosmo(cosmo):
    if isinstance(cosmo, dict):
        c = FlatwCDM(Om0=cosmo["omega_m"], H0=100 * cosmo["h"], w0=cosmo["w0"])
    elif isinstance(cosmo, FLRW):
        c = cosmo
    else:
        TypeError(f"cannot convert {type(cosmo)=} to astropy.cosmology.FLRW object.")

    return c


def lensing_efficiency(z_l, z_s, cosmo):
    """Get lensing efficiency for lens at z_l and source at z_s in cosmo."""
    cosmo = convert_cosmo(cosmo)
    D_s = cosmo.angular_diameter_distance(z_s)
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
    return np.where(D_ls / D_s < 0, 0., D_ls / D_s)


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
    cosmo = convert_cosmo(cosmo)

    # [(M_odot / h) / (Mpc / h)]
    alpha = (constants.c ** 2 / (4 * np.pi * constants.G)).to(u.solMass / u.Mpc)
    sigma_crit = (
        alpha * 1 / (beta_mean * cosmo.angular_diameter_distance(z_l).to(u.Mpc))
    ).to(u.Msun / u.Mpc ** 2)

    if littleh:
        sigma_crit = sigma_crit / cosmo.h * u.littleh

    # surface area grows in comoving units => surface mass density goes down
    if comoving:
        sigma_crit *= 1 / (1 + z_l) ** 2

    return sigma_crit


def shape_noise(
    R_bins,
    z_l,
    cosmo={"omega_m": 0.315, "w0": -1.0, "h": 0.7},
    sigma_e=0.25,
    n_arcmin2=10 * u.arcmin ** -2,
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

    if littleh:
        R_unit = u.Mpc / u.littleh
    else:
        R_unit = u.Mpc

    if comoving:
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z=z_l).to(
            R_unit / u.arcmin, equivalencies=with_H0(cosmo.H0)
        )
    else:
        mpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z=z_l).to(
            R_unit / u.arcmin, equivalencies=with_H0(cosmo.H0)
        )

    theta_bins = R_bins.to(R_unit, equivalencies=with_H0(cosmo.H0)) / mpc_per_arcmin
    theta_centers = R_centers.to(R_unit, equivalencies=with_H0(cosmo.H0)) / mpc_per_arcmin

    # (z, R) array
    N_bins = 2 * np.pi * n_arcmin2 * (np.diff(R_bins) * R_centers)
    sigma_bins = sigma_e / (N_bins) ** 0.5
    return sigma_bins


@u.quantity_input
def sample_gal_pos(
    n_gal: u.arcmin ** -2,
    theta_edges: u.arcmin,
) -> u.Quantity:
    """Sample uniform galaxy positions"""
    dtheta = np.diff(theta_edges)
    theta_lo = theta_edges[:-1]
    theta_hi = theta_edges[1:]

    A = 2 * np.pi * theta_lo * dtheta
    N = (n_gal * A).value.astype(int)
    theta_i = [
        np.random.uniform(t_lo.value, t_hi.value, size=nn) * t_lo.unit
        for t_lo, t_hi, nn in zip(theta_lo, theta_hi, N)
    ]
    return theta_i


@np.vectorize
def Q_zetac(theta, theta1, theta2, thetam):
    if (theta < theta1) | (theta >= thetam):
        return np.zeros_like(theta)
    if (theta >= theta1) & (theta  < theta2):
        return 1 / (np.pi * theta ** 2)
    if (theta >= theta2) & (theta < thetam):
        return 1 / (np.pi * theta ** 2) * 1 / (1 - theta2 ** 2 / thetam ** 2)


def sigma_zetac(theta_edges, theta1, theta2, thetam, sigma_gal, n_gal):
    theta_i = np.concatenate(sample_gal_pos(n_gal, theta_edges))
    if theta_i.size > 0:
        Q_i = Q_zetac(
            theta=theta_i.value,
            theta1=theta1.value,
            theta2=theta2.value,
            thetam=thetam.value,
        ) * theta_i.unit ** -2
    else:
        Q_i = 0.

    delta_zetac_i = sigma_gal / (2 ** 0.5 * n_gal) * np.sum(Q_i ** 2) ** 0.5
    return delta_zetac_i


@u.quantity_input
def sigma_delta_m(
        theta1: Union[u.arcmin, u.Mpc, u.Mpc / u.littleh],
        theta2: Union[u.arcmin, u.Mpc, u.Mpc / u.littleh],
        thetam: Union[u.arcmin, u.Mpc, u.Mpc / u.littleh],
        z_l: float,
        z_s: float,
        sigma_gal: float,
        n_gal: u.arcmin ** -2,
        cosmo: FLRW,
        beta_mean: float = None,
        n_bins: int = 10,
        comoving=False,
) -> u.Quantity:
    """Compute the uncertainty in the aperture mass for zeta_c due to
    background galaxy sampling.

    Parameters
    ----------
    theta1 : u.Quantity
        inner radius for measurement
    theta2 : u.Quantity
        inner radius for control annulus
    thetam : u.Quantity
        outer radius for control annulus
    z_l : float
        redshift of lens
    z_s : float
        redshift of source
    sigma_gal : float
        galaxy shape noise
    cosmo : dict or FLRW
        cosmology
    beta_mean : Optional[float]
        mean lensing efficiency, supersedes z_s
    n_bins : int
        number of bins for the observations
    comoving : bool
        if thetas are passed as lengths, are these lengths comoving?
    """
    if theta1.unit.is_equivalent(theta2.unit) and theta2.unit.is_equivalent(thetam.unit):
        if theta1.unit.is_equivalent(u.Mpc) or theta1.unit.is_equivalent(u.Mpc / u.littleh):
            if comoving:
                len2angle = 1. / cosmo.kpc_comoving_per_arcmin(z_l).to(u.Mpc / u.arcmin)
            else:
                len2angle = 1. / cosmo.kpc_proper_per_arcmin(z_l).to(u.Mpc / u.arcmin)

            theta1 = theta1.to(u.Mpc, equivalencies=with_H0(cosmo.H0)) * len2angle
            theta2 = theta2.to(u.Mpc, equivalencies=with_H0(cosmo.H0)) * len2angle
            thetam = thetam.to(u.Mpc, equivalencies=with_H0(cosmo.H0)) * len2angle

    else:
        raise ValueError("theta1, theta2 and thetam should have consistent units")

    theta_edges = np.linspace(theta1, thetam, n_bins + 1)
    sigma_zc = sigma_zetac(
        theta_edges=theta_edges,
        theta1=theta1,
        theta2=theta2,
        thetam=thetam,
        sigma_gal=sigma_gal,
        n_gal=n_gal,
    )

    if beta_mean is None:
        beta_mean = lensing_efficiency(z_l=z_l, z_s=z_s, cosmo=cosmo)

    sigma_m = (
        np.pi * theta1 ** 2 * sigma_zc
        * cosmo.kpc_comoving_per_arcmin(z=z_l).to(
            u.Mpc / u.arcmin, equivalencies=with_H0(cosmo.H0)
        ) ** 2 * sigma_crit(
            z_l=z_l,
            beta_mean=beta_mean,
            comoving=True,
            littleh=False,
            cosmo=cosmo,
        )
    )
    return sigma_m
