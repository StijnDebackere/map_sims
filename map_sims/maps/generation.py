import time
from typing import List, Callable, Tuple, Optional

import astropy.units as u
import numpy as np
from numpy.random import Generator
from scipy.spatial import KDTree
from tqdm import tqdm

import map_sims.io as io
import map_sims.maps.map_layout as map_layout
import map_sims.maps.observables as obs
import map_sims.maps.tools as map_tools
import map_sims.sims.read_sim as read_sim
import map_sims.utilities as util


def get_map_name(
    save_dir: str,
    slice_axis: int,
    snapshot: int,
    method: str,
    map_thickness: u.Quantity,
    coords_name: str = "",
    map_name_append: str = "",
    downsample: bool = False,
    downsample_factor: float = None,
    full: bool = False,
) -> str:
    save_dir = util.check_path(save_dir)
    if coords_name != "":
        coords_name = f"_{coords_name}"
    if downsample:
        coords_name = f"{coords_name}_downsample_{str(downsample_factor).replace('.', 'p')}"
    if map_thickness.size > 1:
        map_name_append = f"{map_name_append}_nslices_{map_thickness.shape[0]}"
    if full:
        map_name_append = f"{map_name_append}_full"

    map_name = (
        f"{save_dir}/{slice_axis}_maps_{method}{coords_name}"
        f"{map_name_append}_{snapshot:03d}.hdf5"
    )
    return map_name


def coords_to_map_bin(
    coords: u.Quantity,
    map_size: u.Quantity,
    map_pix: int,
    box_size: u.Quantity,
    func: Callable,
    map_center: u.Quantity = None,
    logger: util.LoggerType = None,
    **props,
) -> u.Quantity:
    """Convert the given 2D coordinates to a pixelated map of observable
    func taking props as kwargs.

    Parameters
    ----------
    coords : (n, 2) astropy.units.Quantity
        (x, y) coordinates
    map_size : astropy.units.Quantity
        size of the map
    map_pix : int
        square root of number of pixels in map
    box_size : astropy.units.Quantity
        size of the box
    func : callable
        function that calculates observable for each particle
    map_center : (2,) astropy.units.Quantity, optional
        center of the (x, y) coordinate system
        if None, (0, 0) is assumed as origin
    props : dict of (n, ...) or (1,) arrays
        properties to average, should be the kwargs of func

    Returns
    -------
    mapped : (map_pix, map_pix) astropy.units.Quantity
        Sum_{i in pixel} func(**props_i) / A_pix

    """
    n_pix = map_pix ** 2
    pix_size = map_size / map_pix

    if coords.shape[1] > 2:
        raise ValueError("dimension needs to be along axis 1")
    # convert the coordinates to the pixel coordinate system
    # O: origin
    # x: map_center
    #  ___
    # | x |
    # O---
    if map_center is not None:
        map_origin = map_tools.min_diff(np.atleast_1d(map_center), map_size / 2, box_size)

        # compute the offsets w.r.t the map_origin, taking into account
        # periodic boundary conditions
        coords_origin = map_tools.min_diff(coords, map_origin, box_size)
    else:
        # assume coordinates are already with respect to origin
        coords_origin = coords

    if logger:
        t0 = time.time()
    # get the x and y values of the pixelated maps w.r.t. origin
    x_pix = map_tools.get_coords_slices(
        coords=coords_origin, slice_size=pix_size, slice_axis=0
    )
    y_pix = map_tools.get_coords_slices(
        coords=coords_origin, slice_size=pix_size, slice_axis=1
    )
    pixels = np.concatenate([x_pix[..., None], y_pix[..., None]], axis=-1)

    # slice out only pixel values within the map
    in_map = (x_pix < map_pix) & (x_pix >= 0) & (y_pix < map_pix) & (y_pix >= 0)

    # map (i, j) pixels to 1D pixel id = i * num_pix + j for all the
    # pixels in the map
    pix_ids = map_tools.pixel_to_pix_id(pixels[in_map], map_pix)

    # we will need to associate each function value to the correct pixel
    pix_sort_order = np.argsort(pix_ids)
    # if logger:
    #     t1 = time.time()
    #     logger.debug(f"Dividing coords into pixels took {t1 - t0:.2f}s")
    #     t0 = time.time()

    # unique_ids: all unique pix_ids that contain particles
    # loc_ids: location of each pix_id for the sorted list of pix_ids
    #          e.g. for sorted pix_ids = [0, 0, 0, 1, 1, ..., num_pix_side**2 - 1, ...]
    #          we would get back [0, 3, ...]
    # pix_counts: number of times each unique pix_id appears
    unique_ids, loc_ids, pix_counts = np.unique(
        pix_ids[pix_sort_order], return_index=True, return_counts=True,
    )

    # if logger:
    #     t1 = time.time()
    #     logger.debug(f"Counting particles per pixel took {t1 - t0:.2f}s")
    #     t0 = time.time()

    # filter out properties with single value
    unique_props = {}
    other_props = {}
    for prop, value in props.items():
        value_arr = np.atleast_1d(value)
        if value_arr.shape == (1,):
            unique_props[prop] = value_arr
        elif value_arr.shape[-1] == len(in_map):
            other_props[prop] = value_arr[..., in_map]
        else:
            other_props[prop] = value_arr

    # calculate func for each particle, we already divide by A_pix
    if len(other_props.keys()) == 0:
        # only have unique properties, speed up code by only summing value
        func_values = func(**unique_props) / pix_size ** 2

        pixel_values = np.zeros(n_pix, dtype=float)
        pixel_values[unique_ids] = func_values.value * pix_counts
    else:
        func_values = func(**unique_props, **other_props) / pix_size ** 2

        # sort func_values according to pix_ids
        func_values = func_values[pix_sort_order]

        pixel_values = np.zeros(n_pix, dtype=float)

        # need to also add the final value for pix_id = num_pix**2 - 1
        pix_range = np.concatenate([loc_ids, [len(pix_ids)]])
        # fill each pixel_value with the matching slice along func_values
        pixel_values[unique_ids] = np.array(
            [np.sum(func_values[i:j].value) for i, j in zip(pix_range[:-1], pix_range[1:])]
        )

    # if logger:
    #     t1 = time.time()
    #     logger.debug(f"Putting pixel values into map took {t1 - t0:.2f}s")

    # reshape the array to the map we wanted
    # we get (i, j) array with x_pix along rows and y_pix along columns
    # ensure correct units
    mapped = map_tools.pix_id_array_to_map(pixel_values) * func_values.unit
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
    map_size: u.Quantity,
    map_pix: int,
    box_size: u.Quantity,
    func: Callable,
    map_center: u.Quantity = None,
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
    map_size : astropy.units.Quantity
        size of the map
    map_pix : int
        square root of number of pixels in map
    box_size : astropy.units.Quantity
        size of the box
    func : callable
        function that calculates observable for each particle
    map_center : (2,) astropy.units.Quantity, optional
        center of the (x, y) coordinate system
        if None, (0, 0) is assumed as origin
    n_ngb : int
        number of neighbours to determine SPH kernel size
    props : dict of (..., N) or (1,) arrays
        properties to average, should be the kwargs of func

    Returns
    -------
    mapped : (map_pix, map_pix) astropy.units.Quantity
        Sum_{i in pixel} func(**props_i) / A_pix

    """
    if coords.shape[1] > 2:
        raise ValueError("dimension needs to be along axis 1")

    n_coords = coords.shape[0]
    pix_size = map_size / map_pix

    # compute the offsets w.r.t the map_origin, taking into account
    # periodic boundary conditions
    # O: origin
    # x: map_center
    #  ___
    # | x |
    # O---
    if map_center is not None:
        map_origin = map_tools.min_diff(np.atleast_1d(map_center), 0.5 * map_size, box_size)

        # compute the offsets w.r.t the map_origin, taking into account
        # periodic boundary conditions
        coords_origin = map_tools.min_diff(coords, map_origin, box_size)
    else:
        # assume coords already centered
        coords_origin = coords

    start = time.time()
    tree = KDTree(coords_origin)
    dist, _ = tree.query(coords_origin, k=n_ngb)
    end = time.time()
    if logger:
        logger.debug(
            f"calculating h for {n_ngb=} and n_part={coords_origin.shape[0]} took {end - start:.2f}s"
        )

    # 2 times smoothing length for each particle
    h_max = dist.max(axis=-1) * coords_origin.unit

    mapped = np.zeros((map_pix, map_pix))

    start = time.time()
    for idx, (coord_pix, h_pix) in enumerate(
        zip(coords_origin / pix_size, h_max / pix_size)
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
                (k, v[idx]) if np.atleast_1d(v).shape[0] == n_coords
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


def save_map_full(
    sim_suite: str,
    sim_dir: str,
    snapshot: int,
    slice_axis: int,
    box_size: u.Quantity,
    map_pix: int,
    map_thickness: u.Quantity,
    map_types: List[str],
    save_dir: str,
    map_name_append: str = "",
    overwrite: bool = False,
    method: str = "bin",
    n_ngb: int = 30,
    iterate_files: bool = True,
    scramble_files: bool = True,
    save_num_files: int = 50,
    verbose: bool = False,
    logger: util.LoggerType = None,
    rng: Optional[Generator] = None,
    **kwargs,
) -> str:
    """

    """
    t0 = time.time()
    # go from thick to thin
    map_thickness = np.sort(np.atleast_1d(map_thickness))[::-1]
    no_slice_axis = np.arange(0, 3) != slice_axis

    # ensure that save_dir exists
    if save_dir is not None:
        save_dir = util.check_path(save_dir)
    else:
        raise ValueError("need to specify save_dir")

    map_name = get_map_name(
        save_dir=save_dir,
        slice_axis=slice_axis,
        snapshot=snapshot,
        method=method,
        map_thickness=map_thickness,
        coords_name="",
        map_name_append=map_name_append,
        downsample=False,
        downsample_factor=None,
        full=True,
    )
    # save initial data
    metadata = {
        "slice_axis": slice_axis,
        "box_size": box_size,
        "map_size": box_size,
        "map_pix": map_pix,
        "map_thickness": map_thickness,
        "snapshot": snapshot,
        "sim_dir": sim_dir,
        "method": method,
    }
    io.dict_to_hdf5(
        fname=map_name,
        data=metadata,
        overwrite=overwrite,
    )

    # select mapping function
    if method == "bin":
        coords_to_map = coords_to_map_bin

    elif method == "sph":
        coords_to_map = coords_to_map_sph

    # BAHAMAS uses too much memory if loading all data at once
    # => offer option to iterate over single files instead
    if iterate_files:
        file_nums = read_sim.get_file_nums(
            sim_suite=sim_suite,
            sim_dir=sim_dir,
            snapshot=snapshot,
        )
        if scramble_files and isinstance(rng, Generator):
            rng.shuffle(file_nums)
    else:
        file_nums = [None]

    if verbose:
        file_nums = tqdm(file_nums, desc="Iterating snapshots")

    # extract all necessary properties for each map_type
    for idx, file_num in enumerate(file_nums):
        for map_type in map_types:
            tm0 = time.time()

            # get the properties from map_type for the simulation
            ptype = obs.MAP_TYPES_OPTIONS[map_type]["ptype"]
            # copy to ensure nothing gets popped inadvertently
            properties = obs.MAP_TYPES_OPTIONS[map_type]["properties"].copy()
            attributes = obs.MAP_TYPES_OPTIONS[map_type].get("attributes", None)
            func  = obs.MAP_TYPES_OPTIONS[map_type]["func"]

            props = read_sim.read_particle_properties(
                sim_suite=sim_suite,
                sim_dir=sim_dir,
                snapshot=snapshot,
                properties=properties,
                ptype=ptype,
                file_num=file_num,
                verbose=verbose,
            )
            attrs = read_sim.read_simulation_attributes(
                sim_suite=sim_suite,
                sim_dir=sim_dir,
                snapshot=snapshot,
                attributes=attributes,
                ptype=ptype,
                file_num=file_num,
                verbose=verbose,
            )

            coords = props.pop("coordinates")
            props = {**props, **attrs}
            if method == "sph":
                props["n_ngb"] = n_ngb

            for idx_l, dl in enumerate(map_thickness):
                tl0 = time.time()
                if dl >= box_size:
                    sl = ()
                else:
                    sl = (
                        (coords[:, slice_axis] < 0.5 * (box_size + dl))
                        & (coords[:, slice_axis] > 0.5 * (box_size  - dl))
                    )

                # slice particle dependent properties
                for prop, val in props.items():
                    if val.shape[0] == coords.shape[0]:
                        props[prop] = val[sl]

                    elif len(val.shape) == 1 and val.shape[0] == 1:
                        props[prop] = val

                coords = coords[sl]
                mp = coords_to_map(
                    coords=coords[:, no_slice_axis],
                    map_size=box_size,
                    map_pix=map_pix,
                    box_size=box_size,
                    func=func,
                    map_center=None,
                    logger=logger,
                    **props,
                )

                if logger:
                    tf1 = time.time()
                    logger.info(f"{idx/len(file_nums)}: {file_num=} {dl=} calculated in {tf1 - tl0:.2f}s")

                # save result
                result = {
                    map_type: {
                        idx_l: mp,
                    }
                }
                # only add result, should not be part of map_name yet so safe to overwrite
                io.dict_to_hdf5(
                    fname=map_name,
                    data=result,
                    overwrite=True,
                )

                if logger:
                    tl1 = time.time()
                    logger.info(f"{idx/len(file_nums)}: {file_num=} {dl=} saved in {tl1 - tl0:.2f}s")

            if logger:
                tm1 = time.time()
                logger.info(f"{idx/len(file_nums)}: {file_num=} {map_type=} finished in {tm1 - tm0:.2f}s")

    if logger:
        t1 = time.time()
        logger.info(
            f"Finished {snapshot=}, {slice_axis=}, {map_thickness=} for {sim_dir=} in {t1 - t0:.2f}s"
        )

    return map_name
