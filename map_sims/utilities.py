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


def get_logger(log_dir: str, fname: str, level: str = "INFO") -> logging.Logger:
    log_fname = f"{log_dir}/{fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.log"

    if level.lower() == "info":
        level = logging.INFO
    elif level.lower() == "debug":
        level = logging.DEBUG
    elif level.lower() == "warning":
        level = logging.WARNING
    elif level.lower() == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_fname,
        filemode="w",
        format="%(asctime)s - %(name)s [%(levelname)s] %(funcName)s - %(message)s",
        level=level,
        force=True,
    )
    # ensure that we have different loggers for each simulation and snapshot
    # in multiprocessing, PID can be the same across snapshots and sim_idx
    logger = logging.getLogger(f"{os.getpid()}")

    return logger


def on_queue(queue, func, *args, **kwargs):
    res = time_this(func, pid=True)(*args, **kwargs)
    queue.put([os.getpid(), res])


def groupby(data, index, bin_edges):
    """Group data by index binned into bins.

    Values outside bin_edges are dropped"""
    if type(index) == type(bin_edges) == u.Quantity:
        bin_ids = np.digitize(index.to_value(bin_edges.unit), bin_edges.value)
    else:
        bin_ids = np.digitize(index, bin_edges)

    return {i: data[bin_ids == i + 1] for i in range(0, len(bin_edges) - 1)}


def apply_grouped(fun, grouped_data, **kwargs):
    return dict(
        (i, fun(gd, **kwargs)) if gd.size > 0 else (i, np.nan)
        for i, gd in grouped_data.items()
    )


def bin_centers(bin_edges, log=False):
    """Return the center position of bin_edges, with bin_edges along axis -1."""
    if len(bin_edges.shape) == 1:
        if log:
            if type(bin_edges) is u.Quantity:
                centers = (
                    (bin_edges[1:] - bin_edges[:-1])
                    / (np.log(bin_edges.value[1:]) - np.log(bin_edges.value[:-1]))
                )

            else:
                centers = (
                    (bin_edges[1:] - bin_edges[:-1])
                    / (np.log(bin_edges[1:]) - np.log(bin_edges[:-1]))
                )
        else:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    elif len(bin_edges.shape) == 2:
        if bin_edges.shape[-1] == 2:
            if log:
                if type(bin_edges) is u.Quantity:
                    centers = (
                        (bin_edges[:, 1] - bin_edges[:, 0])
                        / (np.log(bin_edges.value[:, 1]) - np.log(bin_edges.value[:, 0]))
                    ).reshape(-1)
                else:
                    centers = (
                        (bin_edges[:, 1] - bin_edges[:, 0])
                        / (np.log(bin_edges[:, 1]) - np.log(bin_edges[:, 0]))
                    ).reshape(-1)
            else:
                centers = 0.5 * (bin_edges[:, 0] + bin_edges[:, 1]).reshape(-1)

    return centers


def get_xy_quantiles(x, y, x_bin_edges, qs, log=True):
    """Determine qs quantiles of y(x) in (logarithmic if log is True) bins
    with x_bin_edges."""
    x_bins = bin_centers(x_bin_edges, log=log)

    results = []
    Ns = []
    for q in qs:
        grouped_data = groupby(
            data=y,
            index=x,
            bin_edges=x_bin_edges,
        )
        results.append(
            apply_grouped(
                fun=np.nanquantile,
                grouped_data=grouped_data,
                q=q,
            )
        )
        Ns.append([len(data) for bn, data in grouped_data.items()])

    Ns = np.asarray(Ns)
    return x_bins, results, Ns


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
