from functools import wraps
from pathlib import Path

import astropy.units as u
import h5py
from numba import jit
import numpy as np
import scipy.interpolate as interp


NE_FILE = Path(__file__).parent / 'tables/electron_densities.hdf5'
XRAY_FILE = Path(__file__).parent / 'tables/x_ray_table.hdf5'


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
def n_e(z, T, rho, X, Y, h, m_H=1.6726e-24 * u.g):
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
    h : float
        little h, dimensionless Hubble parameter
    m_H : float
        Hydrogen mass [Default: 1.6726e-24 g]

    Returns
    -------
    ne : array-like
        electron number density [h^2 Mpc^-3]
    """
    n_H = (X * rho / m_H).to(
        u.cm**-3, equivalencies=u.with_H0(100 * h * (u.km / (u.s * u.Mpc)))
    )

    # get ratio of Helium to Hydrogen for given mass abundances
    n_He_n_H = Y / (3.971 * X)
    z = np.tile(z, (n_H.shape[0]))

    with h5py.File(NE_FILE, 'r') as f:
        z_interp = f['Redshift_bins'][:]
        T_interp = f['Temperature_bins'][:] * u.K
        n_H_interp = f['Hydrogen_density_bins'][:] * u.cm**-3
        n_He_n_H_interp = f['Metal_free/Helium_number_ratio_bins'][:]
        ne_nh_interp = f['Metal_free/Electron_density_over_n_h'][:]

        coords_interp = (
            z_interp,
            n_He_n_H_interp,
            T_interp.value,
            n_H_interp.value
            # np.log10(n_He_n_H_interp),
            # np.log10(T_interp.value),
            # np.log10(n_H_interp.value)
        )
        coords = coords_within_range(
            (z, z_interp),
            (n_He_n_H, n_He_n_H_interp),
            (T.to(u.K).value, T_interp.to(u.K).value),
            (n_H.to(u.cm**-3).value, n_H_interp.to(u.cm**-3).value)
            # (np.log10(n_He_n_H), np.log10(n_He_n_H_interp)),
            # (np.log10(T.value), np.log10(T_interp.value)),
            # (np.log10(n_H.value), np.log10(n_H_interp.value))
        )
        ne = interp.interpn(
            points=coords_interp,
            values=ne_nh_interp,
            xi=coords
        ) * n_H

    return ne.to(u.littleh**2 * u.Mpc**-3, equivalencies=u.with_H0(h * 100 * u.km / (u.s * u.Mpc)))


class interpolate:
    def init(self):
        pass

    def load_table(self):
        self.table = h5py.File(XRAY_FILE, 'r')
        self.x_ray = self.table['0.5-2.0keV']['emissivities'][()]
        self.He_bins = self.table['/Bins/He_bins'][()]
        self.missing_elements = self.table['/Bins/Missing_element'][()]

        self.density_bins = self.table['/Bins/Density_bins/'][()]
        self.temperature_bins = self.table['/Bins/Temperature_bins/'][()]
        self.dn = 0.2
        self.dT = 0.1

        self.solar_metallicity = self.table['/Bins/Solar_metallicities/'][()]


@jit(nopython=True)
def find_dx(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(bins[idx_0[i]] - subdata[i])

    return dx_p


# @jit(nopython=True)
def find_idx(subdata, bins, dbins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        # mask = np.abs(bins - subdata[i]) < dbins
        # idx_p[i, :] = np.sort(np.argsort(mask)[-2:])
        idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

    return idx_p


@jit(nopython=True)
def find_idx_he(subdata, bins):
    num_bins = len(bins)
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        # idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

        # When closest to the highest bin, or above the highest bin, return the one but highest bin,
        # otherwise we will select a second bin which is outside the binrange
        bin_below = min(np.argsort(np.abs(bins[bins <= subdata[i]] - subdata[i]))[0], num_bins - 2)
        idx_p[i, :] = np.array([bin_below, bin_below + 1])

    return idx_p


@jit(nopython=True)
def find_dx_he(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(subdata[i] - bins[idx_0[i]]) / (bins[idx_0[i]+1] - bins[idx_0[i]])
        # dx_p1[i] = np.abs(bins[idx_0[i+1]] - subdata[i])

    return dx_p


@jit(nopython=True)
def get_table_interp(dn, dT, dx_T, dx_n, idx_T, idx_n, idx_he, dx_he, x_ray, abundance_to_solar):
    f_n_T_Z = np.zeros(len(idx_n[:, 0]))
    for i in range(len(idx_n[:, 0])):
        #interpolate He
        f_000 = x_ray[0, idx_he[i, 0], :, idx_T[i, 0], idx_n[i,0]]
        f_001 = x_ray[0, idx_he[i, 0], :, idx_T[i, 0], idx_n[i,1]]
        f_010 = x_ray[0, idx_he[i, 0], :, idx_T[i, 1], idx_n[i,0]]
        f_011 = x_ray[0, idx_he[i, 0], :, idx_T[i, 1], idx_n[i,1]]

        f_100 = x_ray[0, idx_he[i, 1], :, idx_T[i, 0], idx_n[i,0]]
        f_101 = x_ray[0, idx_he[i, 1], :, idx_T[i, 0], idx_n[i,1]]
        f_110 = x_ray[0, idx_he[i, 1], :, idx_T[i, 1], idx_n[i,0]]
        f_111 = x_ray[0, idx_he[i, 1], :, idx_T[i, 1], idx_n[i,1]]

        f_00 = f_000 * (1 - dx_he[i]) + f_100 * dx_he[i]
        f_01 = f_001 * (1 - dx_he[i]) + f_101 * dx_he[i]
        f_10 = f_010 * (1 - dx_he[i]) + f_110 * dx_he[i]
        f_11 = f_011 * (1 - dx_he[i]) + f_111 * dx_he[i]

        #interpolate density
        f_n_T0 = (dn - dx_n[i]) / dn * f_00 + dx_n[i] / dn * f_01
        f_n_T1 = (dn - dx_n[i]) / dn * f_10 + dx_n[i] / dn * f_11

        #interpolate temperature
        f_n_T = (dT - dx_T[i]) / dT * f_n_T0 + dx_T[i] / dT * f_n_T1

        #Apply linear scaling for removed metals
        f_n_T_Z_temp = f_n_T[-1]
        for j in range(len(f_n_T) - 1):
            f_n_T_Z_temp -= (f_n_T[-1] - f_n_T[j]) * abundance_to_solar[i, j]

        f_n_T_Z[i] = f_n_T_Z_temp

    return f_n_T_Z


def x_ray_luminosity(
        z, rho, T, masses, hydrogen_mf, helium_mf, carbon_mf, nitrogen_mf,
        oxygen_mf, neon_mf, magnesium_mf, silicon_mf, iron_mf, h,
        m_H=1.6726e-24 * u.g, fill_value=None):
    """Compute the X-ray luminosity for the given particle data.

    Parameters
    ----------
    z : array-like
        redshift of particles
    rho : array-like
        mass density of particles [M_sun/Mpc^3]
    T : array-like
        temperature of particles [K]
    masses : array-like
        mass of particles [M_sun]
    hydrogen_mf : array-like
        mass fraction in Hydrogen
    helium_mf : array-like
        mass fraction in Helium
    carbon_mf : array-like
        mass fraction in Carbon
    nitrogen_mf : array-like
        mass fraction in Nitrogen
    oxygen_mf : array-like
        mass fraction in Oxygen
    neon_mf : array-like
        mass fraction in Neon
    magnesium_mf : array-like
        mass fraction in Magnesium
    silicon_mf : array-like
        mass fraction in Silicon
    iron_mf : array-like
        mass fraction in Iron
    h : float
        little h, dimensionless Hubble parameter
    m_H : float
        Hydrogen mass [Default: 1.6726e-24 g]
    fill_value : optional
        value to fill in for extrapolations

    Returns
    -------
    luminosities : array-like
        particle X-ray luminosities in L_sun

    """
    #Initialise interpolation class
    interp = interpolate()
    interp.load_table()

    # check bounds for nH and T
    n_H = (hydrogen_mf * rho / m_H).to(
        u.cm**-3, equivalencies=u.with_H0(100 * h * (u.km / (u.s * u.Mpc)))
    )

    log10_n_H, log10_T = coords_within_range(
        (np.log10(n_H.value), np.round(interp.density_bins, 1)),
        (np.log10(T.to(u.K).value), np.round(interp.temperature_bins, 1))
    ).T
    #Initialise the emissivity array which will be returned
    emissivities = np.zeros_like(log10_n_H, dtype = float)

    #Create density mask, round to avoid numerical errors
    density_mask = (log10_n_H >= np.round(interp.density_bins.min(), 1)) & (log10_n_H <= np.round(interp.density_bins.max(), 1))
    #Create temperature mask, round to avoid numerical errors
    temperature_mask = (log10_T >= np.round(interp.temperature_bins.min(), 1)) & (log10_T <= np.round(interp.temperature_bins.max(), 1))

    #Combine masks
    joint_mask = density_mask & temperature_mask

    #Check if within density and temperature bounds
    density_bounds = np.sum(density_mask) == density_mask.shape[0]
    temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
    if ~(density_bounds & temperature_bounds):
        #If no fill_value is set, return an error with some explanation
        if fill_value == None:
            print('Temperature or density are outside of the interpolation range and no fill_value is supplied')
            print('Temperature ranges between log(T) = 5 and log(T) = 9.5')
            print('Density ranges between log(n_H) = -8 and log(n_H) = 6')
            print('Set the kwarg "fill_value = some value" to set all particles outside of the interpolation range to "some value"')
            print('Or limit your particle data set to be within the interpolation range')
            raise ValueError
        else:
            emissivities[~joint_mask] = fill_value

    mass_fraction = np.zeros((len(log10_n_H[joint_mask]), 9))

    #get individual mass fraction
    mass_fraction[:, 0] = hydrogen_mf[joint_mask]
    mass_fraction[:, 1] = helium_mf[joint_mask]
    mass_fraction[:, 2] = carbon_mf[joint_mask]
    mass_fraction[:, 3] = nitrogen_mf[joint_mask]
    mass_fraction[:, 4] = oxygen_mf[joint_mask]
    mass_fraction[:, 5] = neon_mf[joint_mask]
    mass_fraction[:, 6] = magnesium_mf[joint_mask]
    mass_fraction[:, 7] = silicon_mf[joint_mask]
    mass_fraction[:, 8] = iron_mf[joint_mask]

    #Find density offsets
    idx_n = find_idx(log10_n_H[joint_mask], interp.density_bins, interp.dn)
    dx_n = find_dx(log10_n_H[joint_mask], interp.density_bins, idx_n[:, 0].astype(int))

    #Find temperature offsets
    idx_T = find_idx(log10_T[joint_mask], interp.temperature_bins, interp.dT)
    dx_T = find_dx(log10_T[joint_mask], interp.temperature_bins, idx_T[:, 0].astype(int))

    #Find element offsets
    #mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    element_masses = [1, 4.0026, 12.0107, 14.0067, 15.999, 20.1797, 24.305, 28.0855, 55.845]

    #Calculate the abundance wrt to solar
    abundances = mass_fraction / np.array(element_masses)
    abundance_to_solar = 1 - abundances / 10**interp.solar_metallicity

    abundance_to_solar = np.c_[abundance_to_solar[:, :-1], abundance_to_solar[:, -2], abundance_to_solar[:, -2], abundance_to_solar[:, -1]] #Add columns for Calcium and Sulphur and add Iron at the end

    #Find helium offsets
    idx_he = find_idx_he(np.log10(abundances[:, 1]), interp.He_bins)
    dx_he = find_dx_he(np.log10(abundances[:, 1]), interp.He_bins, idx_he[:, 0].astype(int))

    emissivities[joint_mask] = get_table_interp(
        interp.dn, interp.dT, dx_T, dx_n, idx_T.astype(int), idx_n.astype(int), idx_he.astype(int),
        dx_he, interp.x_ray, abundance_to_solar[:, 2:]
    )
    luminosities = (
        10**emissivities * u.erg * u.cm**-3 * u.s**-1 * masses  / rho
    )

    return luminosities.to(u.Lsun, equivalencies=u.with_H0(h * 100 * u.km / (u.s * u.Mpc)))
