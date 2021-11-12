import logging
import os
import time

import astropy.units as u
import numpy as np


def get_logger(fname: str, log_level: str) -> logging.Logger:
    if log_level.lower() == "info":
        level = logging.INFO
    elif log_level.lower() == "debug":
        level = logging.DEBUG
    elif log_level.lower() == "warning":
        level = logging.WARNING
    elif log_level.lower() == "critical":
        level = logging.CRITICAL

    log_fname = f"{fname}-{time.strftime('%Y%m%d_%H%M', time.localtime())}.log"
    logging.basicConfig(
        filename=log_fname,
        filemode="w",
        format="%(asctime)s - %(name)s [%(levelname)s] %(funcName)s - %(message)s",
        level=level,
        force=True,
    )
    logger = logging.getLogger(f"{os.getpid()} - {__name__}")
    return logger


def matched_arrays_to_coords(*xi):
    """
    Convert a set of n arrays of shape (i_0, ..., i_n) to a list
    of (i_0*...*i_n, n) coordinates
    """
    # get single matching shape between all arrays
    # we expect sane inputs, i.e. arrays are already reshaped
    # to have empty dimension axes set to 1
    b = np.broadcast(*xi)
    shape = np.array(b.shape)
    arrays = []
    for x in xi:
        arrays.append(
            np.tile(x, (shape / np.array(x.shape)).astype(int)).flatten()
        )

    return np.array(arrays).T


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


def groupby(data, index, bin_edges):
    """Group data by index binned into bin_edges.

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


def bin_centers(bin_edges, axis=-1, log=False):
    """Return the center position of bin_edges, with bin_edges increasing along axis."""
    sl_lo = [slice(None)] * len(bin_edges.shape)
    sl_hi = [slice(None)] * len(bin_edges.shape)

    sl_hi[axis] = slice(1, None)
    sl_lo[axis] = slice(None, -1)

    sl_hi = tuple(sl_hi)
    sl_lo = tuple(sl_lo)
    if log:
        if type(bin_edges) is u.Quantity:
            centers = (
                (bin_edges[sl_hi] - bin_edges[sl_lo])
                / (np.log(bin_edges.value[sl_hi]) - np.log(bin_edges.value[sl_lo]))
            )

        else:
            centers = (
                (bin_edges[sl_hi] - bin_edges[sl_lo])
                / (np.log(bin_edges[sl_hi]) - np.log(bin_edges[sl_lo]))
            )
    else:
        centers = 0.5 * (bin_edges[sl_lo] + bin_edges[sl_hi])

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
