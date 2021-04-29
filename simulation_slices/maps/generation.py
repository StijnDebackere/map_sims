from typing import List

import numpy as np
from tqdm import tqdm

import simulation_slices.maps.observables as obs
import simulation_slices.maps.tools as tools
import simulation_slices.sims.slicing as slicing
import simulation_slices.utilities as util


def coords_to_map(
        coords, map_center, map_size, map_res, box_size, func,
        **props):
    """Convert the given 2D coordinates to a pixelated map of observable
    func taking props as kwargs.

    Parameters
    ----------
    coords : (2, N) array
        (x, y) coordinates
    map_center : (2,) array
        center of the (x, y) coordinate system
    map_size : float
        size of the map in units of box_size
    map_res : float
        resolution of a pixel in units of box_size
    box_size : float
        periodicity of the box
    func : callable
        function that calculates observable for each particle
    props : dict of (..., N) or (1,) arrays
        properties to average, should be the kwargs of func
    num_threads : int
        number of threads to use

    Returns
    -------
    mapped : (map_extent // map_res, map_extent // map_res) array
        Sum_{i in pixel} func(**props_i) / A_pix

    """
    map_res = util.check_slice_size(slice_size=map_res, box_size=map_size)
    num_pix_side = int(map_size // map_res)
    A_pix = map_res**2

    # convert the coordinates to the pixel coordinate system
    map_origin = tools.min_diff(np.atleast_1d(map_center), map_size / 2, box_size)

    # compute the offsets w.r.t the map_origin, taking into account
    # periodic boundary conditions
    coords_origin = tools.min_diff(coords, map_origin.reshape(2, 1), box_size)

    # get the x and y values of the pixelated maps w.r.t. origin
    x_pix = slicing.get_coords_slices(
        coords=coords_origin, slice_size=map_res, slice_axis=0
    )
    y_pix = slicing.get_coords_slices(
        coords=coords_origin, slice_size=map_res, slice_axis=1
    )

    # slice out only pixel values within the map
    in_map = (
        (x_pix < num_pix_side) & (x_pix >= 0)
        & (y_pix < num_pix_side) & (y_pix >= 0)
    )

    # map (i, j) pixels to 1D pixel id = i + j * num_pix for all the
    # pixels in the map
    pix_ids = tools.pixel_to_pix_id([x_pix[in_map], y_pix[in_map]], num_pix_side)

    props = dict(
        [
            (k, v[..., in_map])
            if v.shape[-1] == len(in_map)
            # if v is a single value, apply it for all coords in pixel
            else (k, v * np.ones(in_map.sum()))
            for k, v in props.items()
        ])

    # calculate func for each particle, we already divide by A_pix
    func_values = func(**props) / A_pix

    # now we need to associate each value to the correct pixel
    sort_order = np.argsort(pix_ids)
    func_values = func_values[sort_order]

    # get the location of each pixel for the sorted pixel list
    # e.g. for sorted pix_ids = [0, 0, 0, 1, 1, ..., num_pix_side**2, ...]
    # we would get back [0, 3, ...]
    unique_ids, loc_ids = np.unique(pix_ids[sort_order], return_index=True)

    # need to also add the final value for pix_id = num_pix**2 - 1
    pix_range = np.concatenate([loc_ids, [len(pix_ids)]])

    pixel_values = np.zeros(num_pix_side**2, dtype=float)
    pixel_values[unique_ids] = np.array([
        np.sum(func_values[i:j]) for i, j in zip(pix_range[:-1], pix_range[1:])
    ])

    # reshape the array to the map we wanted
    # we get (j, i) array with x_pix along columns and y_pix along rows
    mapped = np.atleast_1d(pixel_values).reshape(num_pix_side, num_pix_side)
    return mapped


def get_maps(
        centers: np.ndarray,
        slice_dir: str,
        snapshot: int,
        slice_axis: int,
        slice_size: float,
        box_size: float,
        map_size: float,
        map_res: float,
        map_thickness: float,
        map_types: List[str],
        verbose: bool=False) -> np.ndarray:
    """Project map around coord in a box of (map_size, map_size, slice_size)
    in a map of map_res for given map_type.

    Parameters
    ----------
    centers : (N, 3) array
        (x, y, z) coordinates to slice around
    slice_dir : str
        directory of the saved simulation slices
    snapshot : int
        snapshot to look at
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    slice_size : float
        slice thickness in units of box_size
    box_size : float
        size of the simulation box
    map_size : float
        size of the map in units of box_size
    map_res : float
        resolution of the map in units of box_size
    map_thickness : float
        thickness of the map projection in units of box_size
    map_types : ['gas_mass', 'dm_mass', 'stellar_mass', 'bh_mass', 'sz']
        type of map to compute
    verbose : bool
        show progress bar

    Returns
    -------
    map : (len(map_types), map_size // map_res, map_size // map_res)
        pixelated projected mass for each map_type

    """
    # sort maps for speedup from hdf5 caching
    centers = np.atleast_2d(centers).reshape(-1, 3)
    sort_ids = np.argsort(centers[:, slice_axis])
    centers_sorted = centers[sort_ids]

    extent = np.array([
        centers_sorted[:, slice_axis] - map_thickness / 2,
        centers_sorted[:, slice_axis] + map_thickness / 2
    ]).reshape(1, -1)

    slice_ranges = slicing.get_coords_slices(
        coords=extent, slice_axis=0, slice_size=slice_size,
    ).reshape(2, -1).T

    # create index array that cuts out the slice_axis
    no_slice_axis = np.arange(0, 3) != slice_axis
    num_slices = int(box_size // slice_size)

    slice_file = slicing.open_slice_file(
        save_dir=slice_dir, slice_axis=slice_axis,
        slice_size=float(slice_size), snapshot=snapshot,
    )

    maps = []
    map_props = obs.map_types_to_properties(map_types)
    for center, slice_range in zip(centers_sorted, slice_ranges):
        slice_nums = np.arange(slice_range[0], slice_range[1] + 1) % num_slices
        props = slicing.read_slice_file_properties(
            slice_file=slice_file, slice_nums=slice_nums, properties=map_props,
            )

        coord_map = []
        # extract the properties within the map
        for map_type in map_types:
            ptype = obs.MAP_TYPES_OPTIONS[map_type]['ptype']
            coords = props[ptype]['coordinates'][:]
            # slice bounding cylinder for map
            selection = (
                (
                    tools.dist(
                        coords[slice_axis].reshape(1, -1),
                        center[slice_axis].reshape(1, -1), box_size, axis=0)
                    <= map_thickness / 2
                ) & (
                    tools.dist(
                        coords[no_slice_axis],
                        center[no_slice_axis].reshape(2, 1), box_size, axis=0)
                    <= 2**0.5 * map_size / 2
                )
            )

            # only include props for coords within cylinder
            props_map_type = {}
            for prop in props[ptype].keys() - ['coords']:
                if props[ptype][prop].shape[-1] == coords.shape[-1]:
                    props_map_type[prop] = props[ptype][prop][..., selection]
                elif props[ptype][prop].shape == (1,):
                    props_map_type[prop] = props[ptype][prop]
                    continue

                else:
                    raise ValueError(f'{prop} is not scalar or matching len(coords)')

            # ignore sliced dimension
            coords_2d = coords[no_slice_axis][..., selection]
            map_center = center[no_slice_axis]

            mp = coords_to_map(
                coords=coords_2d, map_center=map_center, map_size=map_size,
                map_res=map_res, box_size=box_size,
                func=obs.MAP_TYPES_OPTIONS[map_type]['func'],
                **props_map_type
            )
            # add axis for map_type dimension
            coord_map.append(mp[None])

        # add axis for centers dimension
        maps.append(np.concatenate(coord_map, axis=0)[None])

    # still need to close the slice_file
    slice_file.close()

    # unsort the maps
    maps = np.concatenate(maps, axis=0)[sort_ids.argsort()]
    return maps


def save_maps(
        centers: np.ndarray,
        slice_dir: str,
        snapshot: int,
        slice_axes: List[int],
        slice_size: float,
        box_size: float,
        map_size: float,
        map_res: float,
        map_thickness: float,
        map_types: List[str],
        save_dir: bool=None,
        coords_name: str="",
        verbose: bool=False) -> List[str]:
    """Save projected maps around coords in a box of (map_size, map_size,
    slice_size) in a map of map_res for given map_types.

    Parameters
    ----------
    centers : (N, 3) array
        (x, y, z) coordinates to slice around
    slice_dir : str
        directory of the saved simulation slices
    snapshot : int
        snapshot to look at
    slice_axes : array-like
        axes to slice along [x=0, y=1, z=2]
    slice_size : float
        slice thickness in units of box_size
    box_size : float
        size of the simulation box
    map_size : float
        size of the map in units of box_size
    map_res : float
        resolution of the map in units of box_size
    map_thickness : float
        thickness of the map projection in units of box_size
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
        save_dir = util.check_path(slice_dir).parent / 'maps'
    else:
        save_dir = util.check_path(save_dir)

    all_fnames = []
    for slice_axis in slice_axes:
        maps = get_maps(
                centers=centers, slice_dir=slice_dir, snapshot=snapshot,
                slice_axis=slice_axis, slice_size=slice_size, box_size=box_size,
                map_size=map_size, map_res=map_res, map_thickness=map_thickness,
                map_types=map_types, verbose=False
        )

        for i, map_type in enumerate(map_types):
            fname = f'{save_dir}/{slice_axis}_map_{map_type}_{coords_name}.npz'
            np.savez(
                fname,
                maps=maps[:, i],
                slice_size=slice_size,
                map_size=map_size,
                map_res=map_res,
                map_thickness=map_thickness,
                map_type=map_type,
            )
            all_fnames.append(fname)

    return all_fnames
