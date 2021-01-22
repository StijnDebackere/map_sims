import multiprocessing as mp
from pathlib import Path
import time

import h5py
from mira_titan import ExtractMiraTitan
import numpy as np
from tqdm import tqdm

import simulation_slices.operations as ops
import simulation_slices.utilities as util

import pdb


@util.time_this
def save_slice_data(
        base_dir, grid, box_size, z, datatype='snap',
        slice_axis=0, slice_size=2):
    """For the simulation in base_dir with the specified grid, box_size and z,
    slice the particle data along the x, y, and z directions. Slices
    are saved in the Particles directory.

    Parameters
    ----------
    base_dir : str
        path of the MiraTitanU directory
    grid : str
        Mira-Titan model
    box_size : float
        box size
    z : float
        redshift
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    slice_size : float
        thickness in Mpc of the slices

    Returns
    -------
    saves particles for each slice in the Particles/slices/
    directory

    """
    slice_axis = util.check_slice_axis(slice_axis)
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = box_size // slice_size

    filenames = {
        0: f'x_slice_size_{util.num_to_str(slice_size)}.hdf5',
        1: f'y_slice_size_{util.num_to_str(slice_size)}.hdf5',
        2: f'z_slice_size_{util.num_to_str(slice_size)}.hdf5'
    }

    extractor = ExtractMiraTitan(
        base_dir=base_dir, grid=grid, box_size=box_size, z=z)

    # crude estimate of maximum number of particles in each slice
    max_size = int(2 * extractor._Ncbrt**3 / num_slices)

    # get the filename to save to
    save_dir = util.check_path(
        extractor.datatype_info[datatype]['data_dir'] / 'slices')
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / filenames[slice_axis]

    # ensure that we start with a clean slate
    print(f'Removing {filename.name} if it exists!')
    filename.unlink(missing_ok=True)

    # create hdf5 file and generate the required datasets
    h5file = h5py.File(str(filename), mode='a')

    h5file.attrs['slice_axis'] = slice_axis
    h5file.attrs['slice_size'] = slice_size
    h5file.attrs['box_size'] = box_size


    for i in range(num_slices):
        h5file.create_dataset(f'{i}/coords', shape=(3, 0), dtype=float, maxshape=(3, max_size))
        h5file.create_dataset(f'{i}/ids', shape=(0,), dtype=int, maxshape=(max_size,))

    # now loop over all snapshot files and add their particle info
    # to the correct slice
    for num in tqdm(
            extractor.datatype_info[datatype]['nums'],
            desc='Slicing particle files'):
        particles = extractor.read_properties(
            datatype=datatype, num=num, props=[
                'x', 'y', 'z', 'id'
            ]
        )
        coords = np.array([
            particles['x'],
            particles['y'],
            particles['z']
        ])
        particle_mass = extractor.simulation_info['m_p']

        slice_dict = ops.slice_particle_list(
            box_size=extractor.box_size,
            slice_size=slice_size,
            slice_axis=slice_axis,
            properties={
                'coords': coords,
                'ids': particles['id'],
            }
        )

        # append results to hdf5 file
        for idx, (coord, i) in enumerate(zip(
                slice_dict['coords'], slice_dict['ids'])):
            if coord:
                dset_coords = h5file[f'{idx}/coords']
                dset_coords.resize(
                    dset_coords.shape[-1] + coord[0].shape[-1], axis=1)
                dset_coords[..., -coord[0].shape[-1]:] = coord[0]

                dset_ids = h5file[f'{idx}/ids']
                dset_ids.resize(
                    dset_ids.shape[-1] + i[0].shape[-1], axis=0)
                dset_ids[..., -i[0].shape[-1]:] = i[0]

    h5file.close()
