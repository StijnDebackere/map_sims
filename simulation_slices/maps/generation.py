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


def get_map(
        coord, slice_dir, snapshot, slice_axis, slice_size, box_size,
        map_size, map_res, map_thickness, map_types, verbose=False):
    """Project map around coord in a box of (map_size, map_size, slice_size)
    in a map of map_res for given map_type.

    Parameters
    ----------
    coord : (3,) array
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
    # determine which slices need to be included
    thickness = np.zeros((3,), dtype=float)
    thickness[slice_axis] += map_thickness / 2
    num_slices = int(box_size // slice_size)

    extent = np.array([
        coord - thickness, coord + thickness
    ]).T

    slice_ids = slicing.get_coords_slices(
        coords=extent, slice_axis=slice_axis, slice_size=slice_size,
    )

    # create index array that cuts out the slice_axis
    no_slice_axis = np.arange(coord.shape[0]) != slice_axis

    maps = {}
    map_props = obs.map_types_to_properties(map_types)
    for idx in range(slice_ids[0], slice_ids[1] + 1):
        idx_mod = idx % num_slices
        props = slicing.read_slice_file_properties(
            properties=map_props, save_dir=slice_dir,
            slice_axis=slice_axis, slice_size=slice_size,
            snapshot=snapshot, slice_num=idx_mod
        )

        # all map_types for slice idx will be added to slice_maps
        slice_maps = []
        for map_type in map_types:
            coords = props[obs.MAP_TYPES_OPTIONS[map_type]['ptype']]['coords']
            # slice bounding cylinder for map
            selection = (
                (
                    tools.dist(
                        coords[slice_axis].reshape(1, -1),
                        coord[slice_axis].reshape(1, -1), box_size, axis=0)
                    <= map_thickness / 2
                ) & (
                    tools.dist(
                        coords[no_slice_axis],
                        coord[no_slice_axis].reshape(2, 1), box_size, axis=0)
                    <= 2**0.5 * map_size / 2
                )
            )

            # only include props for coords within cylinder
            for prop in obs.MAP_TYPES_OPTIONS[map_type]['dsets']:
                if props[prop].shape[-1] == coords.shape[-1]:
                    props[prop] = props[prop][selection]
                elif props[prop].shape == (1,):
                    props[prop] = np.ones(selection.sum()) * props[prop]

                else:
                    raise ValueError(f'{prop} is not scalar or matching len(coords)')

            # ignore sliced dimension
            coords_2d = coords[no_slice_axis][:, selection]
            map_center = coord[no_slice_axis]

            mp = coords_to_map(
                coords=coords_2d, map_center=map_center, map_size=map_size,
                map_res=map_res, box_size=box_size,
                func=obs.MAP_TYPES_OPTIONS[map_type]['func'],
                **props
            )

            slice_maps.append(mp[None, ...])

        # append (len(map_types), n_pix, n_pix) array to maps
        maps.append(np.concatenate(slice_maps, axis=0))

    # sum over all slices
    maps = np.sum(maps, axis=0)
    return maps


def save_maps(
        coords, slice_dir, snapshot, slice_axes, slice_size, box_size,
        map_size, map_res, map_thickness, map_types, save_dir=None, coords_name=""):
    """Save projected maps around coords in a box of (map_size, map_size,
    slice_size) in a map of map_res for given map_types.

    Parameters
    ----------
    coords : (3, N) array
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

    for slice_axis in slice_axes:
        maps = []
        for coord in coords:
            maps.append(
                get_map(
                    coord=coord, slice_dir=slice_dir, snapshot=snapshot,
                    slice_axes=slice_axis, slice_size=slice_size, box_size=box_size,
                    map_size=map_size, map_res=map_res, map_thickness=map_thickness,
                    map_types=map_types, verbose=False)[None, ...]
            )

        maps = np.concatenate(maps, axis=0)
        for i, map_type in enumerate(map_types):
            np.savez(
                f'{save_dir}/{slice_axis}_map_{map_type}_{coords_name}.hdf5',
                maps=maps[:, i],
                slice_size=slice_size,
                map_size=map_size,
                map_res=map_res,
                map_thickness=map_thickness,
                map_type=map_type,
            )
