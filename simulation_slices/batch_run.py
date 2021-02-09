from pathlib import Path

from gadget import Gadget
from multiprocessing import Process, Queue
import numpy as np
import os

import simulation_slices.bahamas as bahamas
import simulation_slices.utilities as util


# def put_maps_on_queue(queue, *args, **kwargs):
#     result = util.time_this(
#         bahamas.get_mass_projection_maps, pid=True)(*args, **kwargs)
#     queue.put([os.getpid(), result])

AXIS2STR = {
    0: 'x',
    1: 'y',
    2: 'z',
}


def order_coords(coords, map_thickness, box_size, slice_axis):
    """Order the list of coords such that each cpu accesses independent
    slice_files.

    Parameters
    ----------
    coords : (3, N) array
        coordinates to order
    map_thickness : float
        thickness of the map matching units of box_size
    box_size : float
        size of the box
    slice_axis : int
        coordinate to slice along

    Returns
    -------
    coords_split : list of coords
        coordinates split up in box_size / map_thickness bins
    """
    # divide the box up in independent regions of map_thickness
    bin_edges = np.arange(0, box_size, map_thickness)

    # sort the coords according to slice_axis
    coords_sorted = coords[:, coords[slice_axis].argsort()]
    bin_ids = np.digitize(coords[slice_axis], bin_edges)
    in_bins = np.unique(bin_ids)

    coords_split = [coords[:, bin_ids == idx] for idx in in_bins]
    return coords_split


def map_bahamas_clusters(
        sim_dir, slice_dir, snapshot, slice_axis, slice_size, box_size,
        map_size=10, map_res=0.1, map_thickness=20,
        parttypes=[0, 1, 4], log10_m200m_range=np.array([14.5, 15.])):
    """For the simulation in sim_dir with slices in slice_file,
    generate maps for all haloes within m200m_range."""
    group_info = Gadget(
        model_dir=sim_dir, file_type='subh', snapnum=snapshot, sim='BAHAMAS')

    # gadget units are in 10^10 M_sun / h
    log10_m200m = 10 + np.log10(
        group_info.read_var('FOF/Group_M_Mean200', gadgetunits=True)
    )
    log10_m200c = 10 + np.log10(
        group_info.read_var('FOF/Group_M_Crit200', gadgetunits=True)
    )
    log10_m500c = 10 + np.log10(
        group_info.read_var('FOF/Group_M_Crit500', gadgetunits=True)
    )

    group_ids = np.arange(len(log10_m200m))

    # and the group centers
    # (N, 3) array
    centers = group_info.read_var('FOF/GroupCentreOfPotential', gadgetunits=True)

    selected = (
        (log10_m200m > log10_m200m_range.min())
        & (log10_m200m < log10_m200m_range.max())
    )

    # set up multiprocessing
    out_q = Queue()
    procs = []
    centers_split = order_coords(
        # need to transpose centers to (3, N) array
        coords=centers[selected].T, map_thickness=map_thickness,
        box_size=group_info.boxsize, slice_axis=slice_axis)
    n_cpus = len(centers_split)

    for c in centers_split:
        process = Process(
            target=util.on_queue,
            args=(out_q, bahamas.get_mass_projection_maps),
            kwargs={
                # these are (3, N) arrays
                'coords': c,
                'slice_dir': slice_dir,
                'snapshot': snapshot,
                'slice_axis': slice_axis,
                'slice_size': slice_size,
                'box_size': box_size,
                'map_size': map_size,
                'map_res': map_res,
                'map_thickness': map_thickness,
                'parttypes': parttypes,
                'verbose': False
            }
        )
        procs.append(process)
        process.start()

    results = []
    for _ in range(n_cpus):
        results.append(out_q.get())

    for proc in procs:
        proc.join()

    results.sort()
    maps = np.concatenate([item[1] for item in results], axis=0)
    maps = maps.reshape((-1, len(parttypes)) + maps.shape[-2:])

    fname = (
        Path(slice_dir) / f'{AXIS2STR[slice_axis]}_maps_size_{map_size}_'
        f'res_{map_res}_L_{map_thickness}.npz'
    )

    np.savez(
        fname,
        log10_m200m=log10_m200m[selected],
        log10_m200c=log10_m200c[selected],
        log10_m500c=log10_m500c[selected],
        centers=centers[selected],
        group_ids=group_ids[selected],
        map_size=map_size, map_res=map_res,
        map_thickness=map_thickness,
        snapshot=snapshot,
        maps=maps,
    )
    return maps
