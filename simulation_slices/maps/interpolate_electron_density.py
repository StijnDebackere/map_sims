from functools import wraps
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import scipy.interpolate as interp

import pdb


NE_FILE = Path(__file__).parent / 'electron_densities.hdf5'


def args_to_arrays(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arr_args = [np.atleast_1d(arg) for arg in args]
        arr_kwargs = {key: np.atleast_1d(val) for key, val in kwargs.items()}
        return func(*arr_args, **arr_kwargs)
    return wrapper


def coords_within_range(*args):
    """For each tuple set, ensure that the first argument is within range
    of the second one. Return as a set of coordinates of (ndim, len(args)).

    Parameters
    ----------
    list of 2-tuples

    Examples
    --------
    >>> coords_within_range((np.linspace(0, 10), np.array([3, 8])))
    array([[3],
           [3],
           [3],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8],
           [8]])
    """
    coords = []
    for arg in args:
        mn = arg[1].min()
        mx = arg[1].max()
        arg[0][arg[0] < mn] = mn
        arg[0][arg[0] > mx] = mx
        coords.append(arg[0].reshape(-1, 1))

    return np.concatenate(coords, axis=1)


@args_to_arrays
def n_e(z, T, rho, X, Y, m_H=1.6726e-24):
    """Get the electron number density for a gas at given redshift z,
    temperature T and with Hydrogen/Helium mass fractions of X/Y.

    Parameters
    ----------
    z : array-like
        redshift of particles
    T : array-like
        temperature of particles [K]
    rho : array-like
        mass density of particles [M_sun/Mpc^3]
    X : array-like
        mass fraction in Hydrogen
    Y : array-like
        mass fraction in Helium
    m_H : float
        Hydrogen mass [Default: 1.6726e-24 g]

    Returns
    -------
    ne : array-like
        electron number density [Mpc^-3]
    """
    T *= u.K
    n_H = (X * rho / m_H * u.M_sun / (u.Mpc**3 * u.g)).to(u.cm**-3)
    # get ratio of Helium to Hydrogen for given mass abundances
    n_He_n_H = Y / (3.971 * X)
    z = np.tile(z, (n_H.shape[0]))

    with h5py.File(NE_FILE, 'r') as f:
        z_interp = f['Redshift_bins'][:]
        T_interp = f['Temperature_bins'][:] * u.K
        n_H_interp = f['Hydrogen_density_bins'][:] * u.cm**-3
        n_He_n_H_interp = f['Metal_free/Helium_number_ratio_bins'][:]
        ne_nh_interp = f['Metal_free/Electron_density_over_n_h'][:]

        coords = coords_within_range(
            (z, z_interp), (n_He_n_H, n_He_n_H_interp),
            (T.value, T_interp.value), (n_H.value, n_H_interp.value)
        )
        ne = interp.interpn(
            points=(z_interp, n_He_n_H_interp, T_interp.value, n_H_interp.value),
            values=ne_nh_interp,
            xi=coords
        ) * n_H

    return ne.to(u.Mpc**-3).value
