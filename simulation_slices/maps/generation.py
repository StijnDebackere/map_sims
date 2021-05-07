from typing import List, Callable, Tuple

import astropy.units as u
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

import simulation_slices.maps.observables as obs
import simulation_slices.maps.tools as tools
import simulation_slices.sims.slicing as slicing
import simulation_slices.utilities as util


def kernel(r: u.Quantity, h: u.Quantity, dim: int = 2) -> u.Quantity:
    """
    The Wendland Quintic spline kernel.

    """
    if dim == 2:
        alpha_d = 7 / (4 * np.pi * h * h)
    elif dim == 3:
        alpha_d = 21 / (16 * np.pi * h * h * h)

    q = np.abs(r / h)

    one_minus_half_q = 1 - 0.5 * q
    one_minus_half_q_f = np.power(one_minus_half_q, 4)

    W = alpha_d * one_minus_half_q_f * (2 * q + 1)
    W[q > 2] = 0
    return W


def coords_to_map(
    coords: u.Quantity,
    map_center: u.Quantity,
    map_size: u.Quantity,
    map_pix: int,
    box_size: u.Quantity,
    func: Callable,
    n_ngb: int = 30,
    **props,
) -> u.Quantity:
    """Convert the given 2D coordinates to a pixelated map of observable
    func taking props as kwargs.

    Parameters
    ----------
    coords : (2, N) astropy.units.Quantity
        (x, y) coordinates
    map_center : (2,) astropy.units.Quantity
        center of the (x, y) coordinate system
    map_size : astropy.units.Quantity
        size of the map
    map_pix : int
        square root of number of pixels in map
    box_size : astropy.units.Quantity
        size of the box
    func : callable
        function that calculates observable for each particle
    n_ngb : int
        number of neighbours to determine SPH kernel size
    props : dict of (..., N) or (1,) arrays
        properties to average, should be the kwargs of func

    Returns
    -------
    mapped : (map_pix, map_pix) astropy.units.Quantity
        Sum_{i in pixel} func(**props_i) / A_pix

    """
    n_pix = map_pix ** 2
    pix_size = map_size / map_pix

    # convert the coordinates to the pixel coordinate system
    # O: origin
    # x: map_center
    #  ___
    # | x |
    # O---
    map_origin = tools.min_diff(np.atleast_1d(map_center), map_size / 2, box_size)

    # compute the offsets w.r.t the map_origin, taking into account
    # periodic boundary conditions
    coords_origin = tools.min_diff(coords, map_origin.reshape(2, 1), box_size)

    tree = KDTree(coords_origin.T)
    dist, _ = tree.query(coords_origin.T, k=n_ngb)

    # 2 times smoothing length for each particle
    h_max = dist.max(axis=-1) * coords_origin.unit
    h = 0.5 * h_max

    pix_range = np.linspace(0.5, map_pix - 0.5, map_pix) * pix_size
    # get the pixel centers for the grid
    pixel_centers = util.arrays_to_coords(pix_range, pix_range)

    mapped = np.zeros((map_pix, map_pix))
    # (n_pix, n_coords) array with pairwise distances for each pixel and particle
    dist_grid = np.linalg.norm(pixel_centers[..., None] - coords_origin[None], axis=1)

    # for each of n_pix only include particles within 2 * smoothing length
    included = dist_grid < h_max[None]

    # calculate weight for each particle using the distance to each pixel
    weight = func(**props) * kernel(dist_grid[included], h[np.where(included)[1]])
    pix_values = np.zeros(dist_grid.shape) * weight.unit
    pix_values[included] = weight
    mapped = pix_values.sum(axis=-1).reshape(map_pix, map_pix)

    return mapped


def get_maps(
    centers: u.Quantity,
    slice_dir: str,
    snapshot: int,
    slice_axis: int,
    box_size: u.Quantity,
    num_slices: int,
    map_pix: int,
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_types: List[str],
    verbose: bool = False,
) -> u.Quantity:
    """Project map around coord in a box of (map_size, map_size, map_thickness)
    in a map of (map_pix, map_pix) for given map_type.

    Parameters
    ----------
    centers : (N, 3) astropy.units.Quantity
        (x, y, z) coordinates to slice around
    slice_dir : str
        directory of the saved simulation slices
    snapshot : int
        snapshot to look at
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    box_size : astropy.units.Quantity
        size of simulation
    num_slices : int
        total number of slices
    map_pix : int
        square root of number of pixels in map
    map_size : astropy.units.Quantity
        size of the map
    map_thickness : astropy.units.Quantity
        thickness of the map projection
    map_types : ['gas_mass', 'dm_mass', 'stellar_mass', 'bh_mass', 'sz']
        type of map to compute
    verbose : bool
        show progress bar

    Returns
    -------
    map : (len(map_types), map_pix, map_pix)
        pixelated projected mass for each map_type

    """
    # sort maps for speedup from hdf5 caching
    centers = np.atleast_2d(centers).reshape(-1, 3)
    sort_ids = np.argsort(centers[:, slice_axis])
    centers_sorted = centers[sort_ids]

    lower_bound = centers_sorted[:, slice_axis] - map_thickness / 2
    upper_bound = centers_sorted[:, slice_axis] + map_thickness / 2
    extent = (
        np.array(
            [
                lower_bound.to_value(lower_bound.unit),
                upper_bound.to_value(lower_bound.unit),
            ]
        ).reshape(1, -1)
        * lower_bound.unit
    )

    slice_size = box_size / num_slices
    slice_ranges = (
        slicing.get_coords_slices(
            coords=extent,
            slice_axis=0,
            slice_size=slice_size,
        )
        .reshape(2, -1)
        .T
    )

    # create index array that cuts out the slice_axis
    no_slice_axis = np.arange(0, 3) != slice_axis

    slice_file = slicing.open_slice_file(
        save_dir=slice_dir,
        snapshot=snapshot,
        slice_axis=slice_axis,
        num_slices=num_slices,
    )

    maps = {
        map_type: {
            "maps": [],
            "units": [],
        } for map_type in map_types
    }
    map_props = obs.map_types_to_properties(map_types)
    for center, slice_range in zip(centers_sorted, slice_ranges):
        # need to correct for periodic boundary conditions
        # only do this here because we need arange to work correctly
        slice_nums = np.arange(slice_range[0], slice_range[1] + 1) % num_slices
        props = slicing.read_slice_file_properties(
            slice_file=slice_file,
            slice_nums=slice_nums,
            properties=map_props,
        )

        # extract the properties within the map
        for map_type in map_types:
            ptype = obs.MAP_TYPES_OPTIONS[map_type]["ptype"]
            coords = props[ptype]["coordinates"][:]
            # slice bounding cylinder for map
            selection = (
                tools.dist(
                    coords[slice_axis].reshape(1, -1),
                    center[slice_axis].reshape(1, -1),
                    box_size,
                    axis=0,
                )
                <= map_thickness / 2
            ) & (
                tools.dist(
                    coords[no_slice_axis],
                    center[no_slice_axis].reshape(2, 1),
                    box_size,
                    axis=0,
                )
                <= 2 ** 0.5 * map_size / 2
            )

            # only include props for coords within cylinder
            props_map_type = {}
            for prop in props[ptype].keys() - ["coordinates"]:
                if props[ptype][prop].shape[-1] == coords.shape[-1]:
                    props_map_type[prop] = props[ptype][prop][..., selection]
                elif props[ptype][prop].shape == (1,):
                    props_map_type[prop] = props[ptype][prop]
                    continue

                else:
                    raise ValueError(f"{prop} is not scalar or matching len(coords)")

            # ignore sliced dimension
            coords_2d = coords[no_slice_axis][..., selection]
            map_center = center[no_slice_axis]

            mp = coords_to_map(
                coords=coords_2d,
                map_center=map_center,
                map_size=map_size,
                map_pix=map_pix,
                box_size=box_size,
                func=obs.MAP_TYPES_OPTIONS[map_type]["func"],
                **props_map_type,
            )
            maps[map_type]["maps"].append(mp)
            maps[map_type]["units"].append(str(mp.unit))

    # still need to close the slice_file
    slice_file.close()

    # unsort the maps
    maps = {
        map_type: {
            key: np.asarray(val)[sort_ids.argsort()] for key, val in d.items()
        } for map_type, d in maps.items()
    }
    return maps


def save_maps(
    centers: u.Quantity,
    slice_dir: str,
    snapshot: int,
    slice_axes: List[int],
    num_slices: int,
    box_size: u.Quantity,
    map_pix: int,
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_types: List[str],
    save_dir: bool = None,
    coords_name: str = "",
    verbose: bool = False,
) -> List[str]:
    """Save projected maps around coords in a box of (map_size, map_size,
    slice_size) in a map of map_pix for given map_types.

    Parameters
    ----------
    centers : (N, 3) astropy.units.Quantity
        (x, y, z) coordinates to slice around
    slice_dir : str
        directory of the saved simulation slices
    snapshot : int
        snapshot to look at
    slice_axes : array-like
        axes to slice along [x=0, y=1, z=2]
    num_slices : int
        total number of slices
    box_size : astropy.units.Quantity
        size of the simulation box
    map_pix : int
        resolution of the map
    map_size : astropy.units.Quantity
        size of the map in units of box_size
    map_thickness : astropy.units.Quantity
        thickness of the map projection
    map_types : ['gas_mass', 'dm_mass', 'stellar_mass', 'bh_mass', 'sz']
        type of map to compute
    save_dir : str
        directory to save in
    coords_name : str
        identifier to append to filenames

    Returns
    -------
    saves maps to save_dir

    """
    if save_dir is None:
        save_dir = util.check_path(slice_dir).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    all_fnames = []
    for slice_axis in slice_axes:
        maps = get_maps(
            centers=centers,
            slice_dir=slice_dir,
            snapshot=snapshot,
            slice_axis=slice_axis,
            num_slices=num_slices,
            box_size=box_size,
            map_size=map_size,
            map_pix=map_pix,
            map_thickness=map_thickness,
            map_types=map_types,
            verbose=False,
        )

        for map_type in map_types:
            fname = f"{save_dir}/{slice_axis}_map_{map_type}_{coords_name}_{snapshot:03d}.npz"
            np.savez(
                fname,
                maps=maps[map_type]["maps"],
                map_units=maps[map_type]["units"],
                centers=centers,
                snapshot=snapshot,
                slice_axis=slice_axis,
                length_units=str(box_size.unit),
                box_size=box_size,
                map_size=map_size,
                map_thickness=map_thickness,
                map_pix=map_pix,
                map_type=map_type,
            )
            all_fnames.append(fname)

    return all_fnames
