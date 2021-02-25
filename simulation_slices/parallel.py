from itertools import repeat
from multiprocessing import Pool


def starmap_with_kwargs(pool, fn, kwargs_iter):
    kwargs_for_starmap = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_kwargs, kwargs_for_starmap)


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


def compute_tasks(fn, n_workers, kwargs_iter):
    with Pool(n_workers) as pool:
        result = starmap_with_kwargs(pool, fn, kwargs_iter)

    return result
