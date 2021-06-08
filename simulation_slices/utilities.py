from functools import wraps
import logging
import os
import pathlib
from pathlib import Path
import time
from typing import Union, List

import astropy.units as u
from dagster import DagsterLogManager
import numpy as np


LoggerType = Union[logging.Logger, DagsterLogManager]


def on_queue(queue, func, *args, **kwargs):
    res = time_this(func, pid=True)(*args, **kwargs)
    queue.put([os.getpid(), res])


def groupby(data, index, bins):
    """Group data by index binned into bins.

    Values outside bins are dropped"""
    if type(index) == type(bins) == u.Quantity:
        bin_ids = np.digitize(index.to_value(bins.unit), bins.value)
    else:
        bin_ids = np.digitize(index, bins)

    return {i: data[bin_ids == i + 1] for i in range(0, len(bins) - 1)}


def apply_grouped(fun, grouped_data, **kwargs):
    return {i: fun(gd, **kwargs) for i, gd in grouped_data.items()}


def bin_centers(bins, log=False):
    """Return the center position of bins, with bins along axis -1."""
    if log:
        if type(bins) is u.Quantity:
            centers = (
                (bins[..., 1:] - bins[..., :-1])
                / (np.log(bins.value[..., 1:]) - np.log(bins.value[..., :-1]))
            )

        else:
            centers = (
                (bins[..., 1:] - bins[..., :-1])
                / (np.log(bins[..., 1:]) - np.log(bins[..., :-1]))
            )
    else:
        centers = 0.5 * (bins[..., :-1] + bins[..., 1:])

    return centers


def arrays_to_coords(*xi):
    """
    Convert a set of N 1-D coordinate arrays to a regular coordinate grid of
    dimension (npoints, N) for the interpolator
    """
    # the meshgrid matches each of the *xi to all the other *xj
    Xi = np.meshgrid(*xi, indexing='ij')

    # now we create a column vector of all the coordinates
    coords = np.concatenate([X.reshape(X.shape + (1,)) for X in Xi], axis=-1)

    return coords.reshape(-1, len(xi))


def coord_ids_to_curve(coord_ids, coord_shape):
    """Convert a list of coord_ids = (x, y, z) with maximum coord_shape = (n_x, n_y, n_z)
    a space-filling curve index.

    e.g. in 3D:
    curve_id = x + y * n_x + z * n_x * n_y

    Parameters
    ----------
    coord_ids : (n, dim) array-like
        index along each coordinate axis
    coord_shape : (dim,) array-like
        maximum index along each coordinate axis

    Returns
    -------
    curve_ids : (n,) array-like
        index along curve for each coord_id

    """
    coord_ids = np.atleast_2d(coord_ids)
    coord_shape = np.atleast_1d(coord_shape)
    if coord_ids.dtype != int:
        raise ValueError("coord_ids should be int")
    if coord_ids.shape[1] != coord_shape.shape[0]:
        raise ValueError("axis 1 of coord_ids should match axis 0 of coord_shape")

    # coord_ids => (n, i)
    # coord_shape => (i, )
    dim_factor = np.concatenate([[1], np.cumprod(coord_shape[:-1])]).astype(int)
    curve_ids = np.sum(coord_ids * dim_factor, axis=-1)

    return curve_ids


def curve_to_coord_ids(curve_ids, coord_shape):
    """Convert a list of curve ids into coord_ids = (x, y, z) with maximum
    coord_shape = (n_x, n_y, n_z).

    In 3D:
    curve_id = x + y * n_x + z * n_x * n_y
    => coord_id = (x, y, z)

    e.g. curve_id = 954 for coord_shape = [10, 15, 20]
    => [[ 954 % (15 * 10) = 54 => 54 % 10 = 4 => 4 // 1 = 4 ]
        [ 954 % (15 * 10) = 54 => 54 // 10 = 5 ]
        [ 954 // (15 * 10) = 6 ]]
    => coord_id = [4, 5, 6]

    Parameters
    ----------
    curve_ids : (n,) array-like
        index along curve for each coord_id
    coord_shape : (dim,) array-like
        maximum index along each coordinate axis

    Returns
    -------
    coord_ids : (n, dim) array-like
        index along each coordinate axis

    """
    curve_ids = np.atleast_1d(curve_ids)
    coord_shape = np.atleast_1d(coord_shape)
    if len(curve_ids.shape) > 1:
        raise ValueError("curve_ids should be 1D array")
    if len(coord_shape.shape) > 1:
        raise ValueError("coord_shape should be 1D array")

    # will construct coord_ids from long division into coord_shape
    coord_ids = np.tile(curve_ids[..., None], (1, coord_shape.shape[0]))

    # increment in number of elements for increment in coord_id along each dimension
    # e.g. coord_shape = [10, 15, 20]
    # => [1, 10, 150]
    dim_factor = np.concatenate([[1], np.cumprod(coord_shape[:-1])])

    for dim in range(0, coord_shape.shape[0]):
        # what is remainder of coord_id until dimension
        for div in dim_factor[dim + 1:][::-1]:
            coord_ids[:, dim] = coord_ids[:, dim] % div

        # what is long_division along dimension
        coord_ids[:, dim] = coord_ids[:, dim] // dim_factor[dim]

    return coord_ids


def get_subvolume_ranges(
    box_size: u.Quantity,
    n_divides: List[int],
    n_sub: int = 100,
    curve_ids: List[int] = None,
):
    """Divide simulation of box_size into n_divides along axes,
    choose n_sub volumes.

    Parameters
    ----------
    box_size : astropy.units.Quantity
        box size
    n_divides : int or (3,) int array-like
        number of divisions of L along each dimension
    n_sub : int
        number of subvolume ranges to return
    curve_ids : array-like or None
        choice of curve_ids to get ranges for

    Returns
    -------
    coord_ranges : (n_sub, 3, 2) astropy.units.Quantity
        coordinate range for each n_sub
    curve_ids : (n_sub,) array-like
        unique space-filling curve id for each subvolume
    """
    n_divides = np.atleast_1d(n_divides).reshape(-1)
    delta_L = box_size / n_divides

    if n_divides.shape[0] == 1:
        n_divides = np.repeat(n_divides, 3)

    if n_divides.shape[0] != 3:
        raise ValueError("need n_divides for 3 dimensions.")

    if curve_ids is not None:
        coord_ids = curve_to_coord_ids(curve_ids=curve_ids, coord_shape=n_divides)
        coord_ranges = np.zeros((len(curve_ids), 3, 2), dtype=float) * delta_L.unit

    else:
        # we want non-overlapping subvolumes
        all_coord_ids = arrays_to_coords(*[np.arange(0, ndiv) for ndiv in n_divides])
        chosen_ids = np.random.choice(all_coord_ids.shape[0], replace=False, size=n_sub)

        coord_ids = all_coord_ids[chosen_ids]
        curve_ids = coord_ids_to_curve(coord_ids, n_divides)
        coord_ranges = np.zeros((n_sub, 3, 2), dtype=float) * delta_L.unit


    for idx, coord_id in enumerate(coord_ids):
        coord_range = np.array([coord_id * delta_L, (coord_id + 1) * delta_L]).T
        coord_ranges[idx] = coord_range * delta_L.unit

    return coord_ranges, curve_ids


def time_this(pid=False, logger=None):
    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t1 = time.time()
            result = func(*args, **kwargs)
            t2 = time.time()
            if pid and not logger:
                message = (
                    f'Process {os.getpid()} - '
                    f'evaluating {func.__name__} took {t2 - t1:.2f}s'
                )
            else:
                message = (
                    f'evaluating {func.__name__} took {t2 - t1:.2f}s'
                )

            if logger:
                logger.debug(message)
            else:
                print(message)

            return result
        return wrapper
    return outer


def check_slice_axis(slice_axis):
    """Check whether slice_axis is either 0, 1, or 2."""
    if slice_axis not in [0, 1, 2]:
        raise ValueError('slice_axis should be either 0, 1, or 2')
    return slice_axis


def check_path(path):
    """Ensure path is a pathlib.PosixPath."""
    if not type(path) is pathlib.PosixPath:
        path = Path(path)

    if not path.exists():
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    return path


def num_to_str(num, unit=None, log=False, precision=3):
    """Convert num to a formatted string with precision, converted to
    unit and with all '.' replaced by 'p'."""
    units = {
        None: 1,
        'd': 10,
        'c': 100,
        'k': 1000,
        'M': 1e6,
        'G': 1e9,
        'T': 1e12,
        'P': 1e15
    }
    if unit not in units.keys():
        raise ValueError(f'unit should be in {units.keys()}')
    if log:
        n = np.log10(num) / units[unit]
    else:
        n = num / units[unit]

    if n % 1 == 0:
        significand = ''
    else:
        significand = f'p{format(n % 1, f".{precision}f").replace("0.", "")}'

    res = f'{format(n // 1, ".0f")}{significand}{unit}'.replace('None', '')

    return res


def make_dir(path):
    """Creates path."""
    dr = check_path(path)
    dr.mkdir(parents=True, exist_ok=True)
