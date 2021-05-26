import time
from typing import List, Callable, Tuple, Optional

import astropy.units as u
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

import simulation_slices.io as io
import simulation_slices.maps.map_layout as map_layout
import simulation_slices.maps.observables as obs
import simulation_slices.maps.tools as tools
import simulation_slices.sims.slicing as slicing
import simulation_slices.utilities as util


def coords_to_map_bin(
    coords: u.Quantity,
    map_center: u.Quantity,
    map_size: u.Quantity,
    map_pix: int,
    box_size: u.Quantity,
    func: Callable,
    logger: util.LoggerType = None,
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

    # get the x and y values of the pixelated maps w.r.t. origin
    x_pix = slicing.get_coords_slices(
        coords=coords_origin, slice_size=pix_size, slice_axis=0
    )
    y_pix = slicing.get_coords_slices(
        coords=coords_origin, slice_size=pix_size, slice_axis=1
    )

    # slice out only pixel values within the map
    in_map = (x_pix < map_pix) & (x_pix >= 0) & (y_pix < map_pix) & (y_pix >= 0)

    # map (i, j) pixels to 1D pixel id = i * num_pix + j for all the
    # pixels in the map
    pix_ids = tools.pixel_to_pix_id([x_pix[in_map], y_pix[in_map]], map_pix)

    props = dict(
        [
            (k, v[..., in_map]) if np.atleast_1d(v).shape[-1] == len(in_map)
            # if v is a single value, apply it for all coords in pixel
            else (k, v * np.ones(in_map.sum()))
            for k, v in props.items()
        ]
    )

    # calculate func for each particle, we already divide by A_pix
    func_values = func(**props) / pix_size ** 2

    # now we need to associate each value to the correct pixel
    sort_order = np.argsort(pix_ids)
    func_values = func_values[sort_order]

    # get the location of each pixel for the sorted pixel list
    # e.g. for sorted pix_ids = [0, 0, 0, 1, 1, ..., num_pix_side**2, ...]
    # we would get back [0, 3, ...]
    unique_ids, loc_ids = np.unique(pix_ids[sort_order], return_index=True)

    # need to also add the final value for pix_id = num_pix**2 - 1
    pix_range = np.concatenate([loc_ids, [len(pix_ids)]])

    pixel_values = np.zeros(n_pix, dtype=float)
    pixel_values[unique_ids] = np.array(
        [np.sum(func_values[i:j].value) for i, j in zip(pix_range[:-1], pix_range[1:])]
    )

    # reshape the array to the map we wanted
    # we get (i, j) array with x_pix along rows and y_pix along columns
    # ensure correct units
    mapped = tools.pix_id_array_to_map(pixel_values) * func_values.unit
    return mapped


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


def coords_to_map_sph(
    coords: u.Quantity,
    map_center: u.Quantity,
    map_size: u.Quantity,
    map_pix: int,
    box_size: u.Quantity,
    func: Callable,
    n_ngb: int = 30,
    logger: util.LoggerType = None,
    **props,
) -> u.Quantity:
    """Smooth the given 2D coordinates to a pixelated map of observable
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
    n_coords = coords.shape[-1]
    pix_size = map_size / map_pix

    # compute the offsets w.r.t the map_origin, taking into account
    # periodic boundary conditions
    # O: origin
    # x: map_center
    #  ___
    # | x |
    # O---
    map_origin = tools.min_diff(np.atleast_1d(map_center), 0.5 * map_size, box_size)

    # compute the offsets w.r.t the map_origin, taking into account
    # periodic boundary conditions
    coords_origin = tools.min_diff(coords, map_origin.reshape(2, 1), box_size)

    start = time.time()
    tree = KDTree(coords_origin.T)
    dist, _ = tree.query(coords_origin.T, k=n_ngb)
    end = time.time()
    if logger:
        logger.debug(
            f"calculating h for {n_ngb=} and n_part={coords_origin.shape[-1]} took {end - start:.2f}s"
        )

    # 2 times smoothing length for each particle
    h_max = dist.max(axis=-1) * coords_origin.unit

    mapped = np.zeros((map_pix, map_pix))

    start = time.time()
    for idx, (coord_pix, h_pix) in enumerate(
        zip(coords_origin.T / pix_size, h_max / pix_size)
    ):
        # all pixels within a smoothing length of particle at coord_pix
        x_lower = np.max([np.floor(coord_pix[0] - h_pix), 0])
        x_upper = np.min([np.floor(coord_pix[0] + h_pix), map_pix - 1]) + 1
        y_lower = np.max([np.floor(coord_pix[1] - h_pix), 0])
        y_upper = np.min([np.floor(coord_pix[1] + h_pix), map_pix - 1]) + 1

        # region around particle completely outside of map
        if x_upper < 0 or y_upper < 0 or x_lower >= map_pix or y_lower >= map_pix:
            continue

        # distance to pixel *centers* (+ 0.5) within h_max of coord_pix
        pix_all = np.mgrid[x_lower:x_upper, y_lower:y_upper].astype(int)
        dist_pix = np.linalg.norm(coord_pix[:, None, None] - (pix_all + 0.5), axis=0)

        # extract the properties for particle at coord_pix
        props = dict(
            [
                (k, v[..., idx]) if np.atleast_1d(v).shape[-1] == n_coords
                # if v is a single value, apply it for all coords in pixel
                else (k, v)
                for k, v in props.items()
            ]
        )

        # smoothing length = 0.5 h_max
        weight = func(**props) * kernel(dist_pix * pix_size, 0.5 * h_pix * pix_size)

        # fill correct pixel values for mapped
        mapped[
            pix_all.reshape(2, -1)[0], pix_all.reshape(2, -1)[1]
        ] += weight.value.flatten()

    unit = weight.unit
    mapped = mapped * unit

    return mapped


def save_maps(
    centers: u.Quantity,
    group_ids: np.ndarray,
    masses: u.Quantity,
    slice_dir: str,
    snapshot: int,
    slice_axis: int,
    num_slices: int,
    box_size: u.Quantity,
    map_pix: int,
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_types: List[str],
    save_dir: str,
    coords_name: str = "",
    map_name_append: str = "",
    overwrite: bool = False,
    swmr: bool = False,
    method: str = None,
    n_ngb: int = 30,
    verbose: bool = False,
    logger: util.LoggerType = None,
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
