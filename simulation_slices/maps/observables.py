import astropy.constants as c
import astropy.units as u

import simulation_slices.maps.interpolate_tables as interp_tables


def particles_masses(masses, **kwargs):
    """Return the list of masses."""
    return masses


def particles_y_sz(
        densities, temperatures, masses,
        electron_number_densities=None, **kwargs):
    """Return the y_SZ for the list of particles."""
    if electron_number_densities is None:
        smoothed_hydrogen = kwargs.pop('smoothed_hydrogen', None)
        smoothed_helium = kwargs.pop('smoothed_helium', None)
        z = kwargs.pop('z', None)
        if z is None or smoothed_hydrogen is None or smoothed_helium is None:
            raise ValueError('z, smoothed_hydrogen and smoothed_helium required as kwargs')

        electron_number_densities = interp_tables.n_e(
            z=z, T=temperatures, rho=densities,
            X=smoothed_hydrogen, Y=smoothed_helium,
        )
    # temperatures are given in K, will divide by pixel area in Mpc^2
    energy_ratio = c.sigma_T * c.k_B / (c.c**2 * c.m_e) * temperatures * u.K
    N_e = masses * electron_number_densities / densities
    return N_e * energy_ratio.to(u.Mpc**2).value


MAP_TYPES_OPTIONS = {
    # no way to work this in cleanly with our remaining tools...
    # combining different keys in func() for coords_to_map is not
    # straigthforward: simply add up all other masses to get this
    # one
    # 'total_mass': {
    #     'keys': ['gas', 'dm', 'stars', 'bh'],
    #     'dsets': ['coords', 'masses'],
    # },
    'gas_mass': {
        'ptype': 'gas',
        'dsets': ['coordinates', 'masses'],
        'func': particles_masses,
    },
    'dm_mass': {
        'ptype': 'dm',
        'dsets': ['coordinates', 'masses'],
        'func': particles_masses,
    },
    'stars_mass': {
        'ptype': 'stars',
        'dsets': ['coordinates', 'masses'],
        'func': particles_masses,
    },
    'bh_mass': {
        'ptype': 'bh',
        'dsets': ['coordinates', 'masses'],
        'func': particles_masses,
    },
    'sz': {
        'ptype': 'gas',
        'dsets': [
            'coordinates', 'masses', 'temperatures', 'densities',
            'electron_number_densities',
        ],
        'func': particles_y_sz,
    },
}


def map_types_to_properties(map_types):
    """Return the properties necessary for each map_type.

    Parameters
    ----------
    map_types : list of str
        one of ['gas_mass', 'dm_mass', 'stars_mass', 'bh_mass', 'sz']

    Returns
    -------
    properties : dict
        dictionary with all required datasets to be loaded for each ptype
        for the given map_types
        - ptype: [dsets]
    """
    if type(map_types) is not list:
        map_types = [map_types]

    valid_map_types = set(map_types) & set(MAP_TYPES_OPTIONS.keys())
    if not valid_map_types:
        raise ValueError(f'{map_types} not in {MAP_TYPES_OPTIONS.keys()}')

    results = {}
    for map_type in valid_map_types:
        ptype = MAP_TYPES_OPTIONS[map_type]['ptype']
        dsets = MAP_TYPES_OPTIONS[map_type]['dsets']

        # only append dsets that are not in results[key] already
        results[ptype] = list(set(dsets) | set(results.get(ptype, [])))

    return results
