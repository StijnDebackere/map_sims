from typing import List

import astropy.constants as c
import astropy.units as u
import numpy as np

import simulation_slices.maps.interpolate_tables as interp_tables
import simulation_slices.utilities as util


def particles_masses(masses, **kwargs):
    """Return the list of masses."""
    return masses


def particles_y_sz(
    densities, temperatures, masses, electron_number_densities=None, **kwargs
):
    """Return the y_SZ for the list of particles."""
    if electron_number_densities is None:
        smoothed_hydrogen = kwargs.pop("smoothed_hydrogen", None)
        smoothed_helium = kwargs.pop("smoothed_helium", None)
        z = kwargs.pop("z", None)
        h = kwargs.pop("h", None)
        if (
            h is None
            or z is None
            or smoothed_hydrogen is None
            or smoothed_helium is None
        ):
            raise ValueError(
                "z, smoothed_hydrogen and smoothed_helium required as kwargs"
            )

        electron_number_densities = interp_tables.n_e(
            z=z,
            h=h,
            T=temperatures,
            rho=densities,
            smoothed_hydrogen=smoothed_hydrogen,
            smoothed_helium=smoothed_helium,
        )
    # temperatures are given in K, will divide by pixel area in Mpc^2
    energy_ratio = c.sigma_T * c.k_B / (c.c ** 2 * c.m_e) * temperatures
    N_e = masses * electron_number_densities / densities
    return N_e * energy_ratio.to(u.Mpc ** 2)


def particles_lum_x_ray(
    z, h, densities, temperatures, masses, **kwargs
):
    """Return the X-ray luminosity for the list of particles."""
    return interp_tables.x_ray_luminosity(
        z=z, h=h, rho=densities, T=temperatures, masses=masses,
        hydrogen_mf=kwargs.pop("smoothed_hydrogen"),
        helium_mf=kwargs.pop("smoothed_helium"),
        carbon_mf=kwargs.pop("smoothed_carbon"),
        nitrogen_mf=kwargs.pop("smoothed_nitrogen"),
        oxygen_mf=kwargs.pop("smoothed_oxygen"),
        neon_mf=kwargs.pop("smoothed_neon"),
        magnesium_mf=kwargs.pop("smoothed_magnesium"),
        silicon_mf=kwargs.pop("smoothed_silicon"),
        iron_mf=kwargs.pop("smoothed_iron"),
        fill_value=np.nan,
    )


# for the given map type, define the properties to be loaded from the simulation
# properties and attributes will be extracted into a dictionary
MAP_TYPES_OPTIONS = {
    "gas_mass": {
        "ptype": "gas",
        "properties": ["coordinates", "masses"],
        "func": particles_masses,
    },
    "dm_mass": {
        "ptype": "dm",
        "properties": ["coordinates", "masses"],
        "func": particles_masses,
    },
    "stars_mass": {
        "ptype": "stars",
        "properties": ["coordinates", "masses"],
        "func": particles_masses,
    },
    "bh_mass": {
        "ptype": "bh",
        "properties": ["coordinates", "masses"],
        "func": particles_masses,
    },
    "y_sz": {
        "ptype": "gas",
        "properties": [
            "coordinates",
            "masses",
            "temperatures",
            "densities",
            "smoothed_hydrogen",
            "smoothed_helium",
        ],
        "attributes": ["z", "h"],
        "func": particles_y_sz,
    },
    "lum_x_ray": {
        "ptype": "gas",
        "properties": [
            "coordinates",
            "masses",
            "temperatures",
            "densities",
            "smoothed_hydrogen",
            "smoothed_helium",
            "smoothed_carbon",
            "smoothed_nitrogen",
            "smoothed_oxygen",
            "smoothed_neon",
            "smoothed_magnesium",
            "smoothed_silicon",
            "smoothed_iron",
        ],
        "attributes": ["z", "h"],
        "func": particles_lum_x_ray,
    },
}


def map_types_to_properties(map_types: List[str]) -> dict:
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
        - ptype:
            - properties: [properties]
            - attributes: [attributes]
    """
    if type(map_types) is not list:
        map_types = [map_types]

    valid_map_types = set(map_types) & set(MAP_TYPES_OPTIONS.keys())
    if not valid_map_types:
        raise ValueError(f"{map_types} not in {MAP_TYPES_OPTIONS.keys()}")

    results = {}
    for map_type in valid_map_types:
        ptype = MAP_TYPES_OPTIONS[map_type]["ptype"]

        # create or add to ptype for map_type
        results[ptype] = results.get(ptype, {})

        properties = MAP_TYPES_OPTIONS[map_type].get("properties", [])
        attributes = MAP_TYPES_OPTIONS[map_type].get("attributes", [])

        # only append properties that are not in results[key] already
        results[ptype]["properties"] = list(
            set(properties) | set(results[ptype].get("properties", []))
        )
        results[ptype]["attributes"] = list(
            set(attributes) | set(results[ptype].get("attributes", []))
        )

    return results
