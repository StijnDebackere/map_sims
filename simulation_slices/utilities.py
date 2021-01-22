from functools import wraps
import os
import pathlib
from pathlib import Path
import time

import numpy as np


# # unfortunately, this wrapper does not work when wrapping
# # a function as a target of a Process...
# def on_queue(func, queue):
#     @wraps(func)
#     def get_result(*args, **kwargs):
#         queue.put([os.getpid(), func(*args, **kwargs)])

#     return get_result
def on_queue(queue, func, *args, **kwargs):
    queue.put([os.getpid(), func(*args, **kwargs)])


def time_this(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Evaluating {func.__name__} took {t2 - t1:.2f}s')
        return result
    return wrapper


def check_coords(coords):
    '''Check whether coordinates has shape (3, N).'''
    coords = np.atleast_2d(coords)
    if len(coords.shape) != 2:
        raise ValueError('coords needs to be 2D array')
    if coords.shape[0] != 3:
        raise ValueError('coords needs to have 3 spatial dimensions along axis 0')
    return coords


def check_properties(properties, N):
    '''Check whether the properties kwargs are all either (..., N) or (1,).'''
    invalid_properties = []
    valid_properties = {}
    for k, v in properties.items():
        v_arr = np.atleast_1d(v)
        if v_arr.shape[-1] == N:
            valid_properties[k] = v_arr
        elif len(v_arr.shape) == 1:
            valid_properties[k] = v_arr
        else:
            invalid_properties.append(k)

    if invalid_properties:
        print(f'Dropped invalid properties {invalid_properties}')

    return valid_properties


def check_slice_axis(slice_axis):
    '''Check whether slice_axis is either 0, 1, or 2.'''
    if slice_axis not in [0, 1, 2]:
        raise ValueError('slice_axis should be either 0, 1, or 2')
    return slice_axis


def check_slice_size(slice_size, box_size):
    '''Ensure that slice_size evenly divides box_size.'''
    if not box_size / slice_size % 1 == 0:
        new_slice_size = box_size / (box_size // slice_size)
        print(
            f'box_size {box_size} needs to be an integer multiple'
            f'of slice_size {slice_size}'
        )
        print(f'changing slice_size to {new_slice_size}')
        return new_slice_size

    return slice_size


def check_path(path):
    '''Ensure path is a pathlib.PosixPath.'''
    if not type(path) is pathlib.PosixPath:
        return Path(path)
    return path
