from pathlib import Path

from gadget import Gadget
import h5py
import numpy as np
import os
import time
import toml

from simulation_slices import Config
import simulation_slices.maps.analysis as analysis
import simulation_slices.maps.generation as map_gen
from simulation_slices.parallel import Queue, Worker
import simulation_slices.sims.bahamas as bahamas
import simulation_slices.utilities as util

import pdb


# def analyze_maps(maps, func, **func_kwargs):
#     """Analyze the given maps with func that requires func_kwargs.

#     Parameters
#     ----------
#     maps : (N, n_pix, n_pix) array
#         maps to be analyzed
#     func : callable
#         analysis function (map, **func_kwargs)

#     Returns
#     -------
#     results : (N, ...) array with results
#     """
#     results = []
#     for mp in maps:
#         results.append(func(mp, **func_kwargs))
#     return results


# def get_M_aps(maps, sigma_crit, pix_scale, R1, R2, Rm):
#     """Get M_ap for all maps."""
#     results = analyze_maps(
#         maps=maps, func=analysis.M_ap_from_sigma_map,
#         sigma_crit=1, filt=analysis.filter_zeta, pix_scale=pix_scale,
#         pix_0=np.array([maps.shape[-1] // 2, maps.shape[-1] // 2]),
#         R1=R1, R2=R2, Rm=Rm
#     )
#     return np.asarray(results)


# def run_full_analysis(maps_file, config):
#     """Run the set of given config settings on maps_file."""
#     if type(config) is not dict:
#         try:
#             with open(config, 'r') as f:
#                 config = yaml.load(f)
#         except TypeError:
#             raise TypeError('config should be a dict or a pathname')

#     data = np.load(maps_file)
#     maps = data['maps']
#     log10_m200m = np.atleast_1d(data['log10_m200m'])
#     log10_m200c = np.atleast_1d(data['log10_m200c'])
#     log10_m500c = np.atleast_1d(data['log10_m500c'])
#     group_ids = np.atleast_1d(data['group_ids'])

#     key_options = {
#         'M_ap': {
#             'func': get_M_aps,
#             'maps': maps.sum(axis=1),
#         },
#         'M_star': {
#             'func': get_M_aps,
#             'maps': maps[:, 4],
#         },
#     }

#     results = {
#         'group_ids': group_ids,
#         'log10_m200m': log10_m200m,
#         'log10_m200c': log10_m200c,
#         'log10_m500c': log10_m500c,
#     }
#     for key in config.keys() & key_options.keys():
#         analysis = key_options[key]
#         results[key] = {}

#         results[key]['config'] = config[key]
#         results[key][''] = analysis['func'](
#             analysis['maps'], **config[key])

#     return results


# def default_analysis():
#     """Default analysis settings."""
#     default = {
#         'M_ap': {
#             'sigma_crit': [1., 1., 1., 1.],
#             'R1': [0.5, 1.0, 1.5, 2.0],
#             'R2': [2.5, 2.5, 2.5, 2.5],
#             'Rm': [3.0, 3.0, 3.0, 3.0],
#         },
#         'M_star': {
#             'sigma_crit': [1., 1., 1., 1.],
#             'R1': [0.5, 1.0, 1.5, 2.0],
#             'R2': [2.5, 2.5, 2.5, 2.5],
#             'Rm': [3.0, 3.0, 3.0, 3.0],
#         }
#     }
#     with open('default_analysis.yaml', 'w') as f:
#         yaml.dump(default, f)


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


# def map_bahamas_clusters(
#         sim_dir, slice_dir, snapshot, slice_axis, slice_size, box_size,
#         map_size=10, map_res=0.1, map_thickness=20,
#         ptypes=[0, 1, 4], log10_m200m_range=np.array([14.5, 15.])):
#     """For the simulation in sim_dir with slices in slice_file,
#     generate maps for all haloes within m200m_range."""
#     group_info = Gadget(
#         model_dir=sim_dir, file_type='subh', snapnum=snapshot, sim='BAHAMAS')

#     # gadget units are in 10^10 M_sun / h
#     log10_m200m = 10 + np.log10(
#         group_info.read_var('FOF/Group_M_Mean200', gadgetunits=True)
#     )
#     log10_m200c = 10 + np.log10(
#         group_info.read_var('FOF/Group_M_Crit200', gadgetunits=True)
#     )
#     log10_m500c = 10 + np.log10(
#         group_info.read_var('FOF/Group_M_Crit500', gadgetunits=True)
#     )

#     group_ids = np.arange(len(log10_m200m))

#     # and the group centers
#     # (N, 3) array
#     centers = group_info.read_var('FOF/GroupCentreOfPotential', gadgetunits=True)

#     selected = (
#         (log10_m200m > log10_m200m_range.min())
#         & (log10_m200m < log10_m200m_range.max())
#     )

#     # set up multiprocessing
#     out_q = Queue()
#     procs = []
#     centers_split = order_coords(
#         # need to transpose centers to (3, N) array
#         coords=centers[selected].T, map_thickness=map_thickness,
#         box_size=group_info.boxsize, slice_axis=slice_axis)
#     n_cpus = len(centers_split)

#     for c in centers_split:
#         process = Process(
#             target=util.on_queue,
#             args=(out_q, bahamas.get_mass_projection_maps),
#             kwargs={
#                 # these are (3, N) arrays
#                 'coords': c,
#                 'slice_dir': slice_dir,
#                 'snapshot': snapshot,
#                 'slice_axis': slice_axis,
#                 'slice_size': slice_size,
#                 'box_size': box_size,
#                 'map_size': map_size,
#                 'map_res': map_res,
#                 'map_thickness': map_thickness,
#                 'ptypes': ptypes,
#                 'verbose': False
#             }
#         )
#         procs.append(process)
#         process.start()

#     results = []
#     for _ in range(n_cpus):
#         results.append(out_q.get())

#     for proc in procs:
#         proc.join()

#     results.sort()
#     maps = np.concatenate([item[1] for item in results], axis=0)
#     maps = maps.reshape((-1, len(ptypes)) + maps.shape[-2:])

#     fname = (
#         Path(slice_dir) / f'{AXIS2STR[slice_axis]}_maps_size_{map_size}_'
#         f'res_{map_res}_L_{map_thickness}.npz'
#     )

#     np.savez(
#         fname,
#         log10_m200m=log10_m200m[selected],
#         log10_m200c=log10_m200c[selected],
#         log10_m500c=log10_m500c[selected],
#         centers=centers[selected],
#         group_ids=group_ids[selected],
#         map_size=map_size, map_res=map_res,
#         map_thickness=map_thickness,
#         snapshot=snapshot,
#         maps=maps,
#     )
#     return maps


def save_coords(coords_dir, sim_name, ):
    pass


def load_coords(coords_dir, sim_name):
    filename = ''


def slice_sim(sim_dir, sim_type, snapshots, slice_axes, slice_size, save_dir):
    if sim_type == 'BAHAMAS':
        for snap in np.atleast_1d(snapshots):
            bahamas.save_slice_data(
                base_dir=str(sim_dir), snapshot=snap, slice_axes=slice_axes,
                slice_size=slice_size, save_dir=save_dir
            )

    return (os.getpid(), f'{sim_dir} sliced')


def map_coords(
        snapshots, box_size, coords_file, coords_name,
        slice_dir, slice_axes, slice_size,
        map_types, map_size, map_res, map_thickness, save_dir):
    with h5py.File(str(coords_file), 'r') as h5file:
        coordinates = h5file['coordinates'][:]

    for snap in np.atleast_1d(snapshots):
        maps = map_gen.save_maps(
            coord=coordinates, slice_dir=slice_dir, snapshot=snap,
            slice_axes=slice_axes, slice_size=slice_size, box_size=box_size,
            map_size=map_size, map_res=map_res, map_thickness=map_thickness,
            map_types=map_types, save_dir=save_dir, coords_name=coords_name,
        )

    return (os.getpid(), f'{save_dir} maps saved')


def analyze_map():
    pass


def run_pipeline(
        config_file, n_workers=10,
        sims=True, maps=True, observables=True):
    config = Config(config_file)

    if sims:
        for p in config.slice_paths:
            p.mkdir(parents=True, exist_ok=True)
        # Start by slicing all the simulations
        sims_q_size = len(config.sim_dirs)
        sims_in_q = Queue(maxsize=sims_q_size)
        sims_out_q = Queue(maxsize=sims_q_size)

        for sim_dir, snaps, save_dir in zip(
                config.sim_paths, config.snapshots, config.slice_paths):
            sims_in_q.put(
                sim_dir=sim_dir,
                sim_type=config.sim_type,
                snapshots=snaps,
                slice_axes=config.slice_axes,
                slice_size=config.slice_size,
                save_dir=save_dir
            )

        workers = []
        for _ in range(n_workers):
            worker = Worker(
                task_fn=slice_sim, worker_in_q=sims_in_q, worker_out_q=sims_out_q
            )
            workers.append(worker)
            worker.start()

        while sims_out_q.qsize() < sims_q_size:
            time.sleep(5)

    if maps:
        for p in config.map_paths:
            p.mkdir(parents=True, exist_ok=True)
        # all simulations should have been sliced now
        # let's generate the required maps
        maps_q_size = len(config.sim_dirs)
        maps_in_q = Queue(maxsize=maps_q_size)
        maps_out_q = Queue(maxsize=maps_q_size)

        for map_dir, slice_dir in zip(config.map_paths, config.slice_paths):
            maps_in_q.put(
                snapshots=config.snapshots,
                box_size=config.box_size,
                coords_file=config.coords_file,
                coords_name=config.coords_name,
                slice_dir=slice_dir,
                slice_axes=config.slice_axes,
                slice_size=config.slice_size,
                map_types=config.map_types,
                map_size=config.map_size,
                map_res=config.map_res,
                map_thickness=config.map_thickness,
                save_dir=save_dir
            )

        workers = []
        for _ in range(n_workers):
            worker = Worker(
                task_fn=map_coords, worker_in_q=maps_in_q, worker_out_q=maps_out_q
            )
            workers.append(worker)
            worker.start()

        while maps_out_q.qsize() < maps_q_size:
            time.sleep(5)

    if observables:
        pass
