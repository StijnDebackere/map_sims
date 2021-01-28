from gadget import Gadget
from multiprocessing import Process, Queue
import numpy as np
import os

import simulation_slices.bahamas as bahamas


def put_maps_on_queue(queue, *args, **kwargs):
    result = bahamas.get_mass_projection_maps(*args, **kwargs)
    queue.put([os.getpid(), result])


def map_bahamas_clusters(
        sim_dir, slice_file, snapshot, n_cpus,
        map_size=10, map_res=0.1, map_thickness=20,
        parttypes=[0, 1, 4], log10_m200m_range=np.array([14.5, 15.])):
    """For the simulation in sim_dir with slices in slice_file,
    generate maps for all haloes within m200m_range."""
    group_info = Gadget(
        model_dir=sim_dir, file_type='subh', snapshot=snapshot, sim='BAHAMAS')

    # gadget units are in 10^10 M_sun / h
    log10_m200m = 10 + np.log10(
        group_info.read_var('FOF/Group_M_Mean200', gadgetunits=True)
    )
    centers = group_info.read_var('FOF/GroupCentreOfPotential', gadgetunits=True)

    selected = (
        (log10_m200m > log10_m200m_range.min())
        & (log10_m200m < log10_m200m_range.max())
    )

    # set up multiprocessing
    out_q = Queue()
    procs = []
    centers_split = np.array_split(centers[selected], n_cpus)

    for c in centers_split:
        process = Process(
            target=put_maps_on_queue,
            args=(out_q),
            kwargs={
                'coords': c,
                'slice_file': slice_file,
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
    return maps
