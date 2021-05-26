from functools import wraps
import os
import pathlib
from pathlib import Path
import time

import astropy.units as u
import numpy as np


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


def join_dict_arrays(a, b, axis=-1):
    """Concatenate with matching keys."""
    if a == {} or b == {}:
        return a or b

    if not a.keys() == b.keys():
        raise ValueError('a and b should have matching keys')

    for key in a.keys():
        a[key] = np.concatenate([a[key], b[key]], axis=axis)

    return a


def time_this(func, pid=False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        if pid:
            print(
                f'Process {os.getpid()}: '
                f'Evaluating {func.__name__} took {t2 - t1:.2f}s'
            )
        else:
            print(f'Evaluating {func.__name__} took {t2 - t1:.2f}s')
        return result
    return wrapper


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
        path.mkdir(parents=True, exist_ok=True)

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
