from gadget import Gadget
import h5py
import numpy as np
from tqdm import tqdm

import simulation_slices.operations as ops
import simulation_slices.utilities as util


@util.time_this
def save_slice_data(
        base_dir, snapshot, datatype='snap', parttypes=[0, 1, 4, 5],
        slice_axis=0, slice_size=2, save_dir=None):
    """For snapshot of simulation in base_dir, slice the particle data for
    all parttypes along the x, y, and z directions. Slices are saved
    in the Snapshots directory by default.

    Parameters
    ----------
    base_dir : str
        path of the MiraTitanU directory
    datatype : str
        particles or snap, particle data to slice
    snapshot : int
        snapshot to look at
    parttypes : iterable of ints
        particle types to read in
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    slice_size : float
        thickness in Mpc of the slices
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/slices/

    Returns
    -------
    saves particles for each slice in the snapshot_xxx/slices/
    directory

    """
    snap_info = Gadget(
        model_dir=base_dir, file_type=datatype, snapnum=snapshot, sim='BAHAMAS')

    box_size = snap_info.boxsize
    slice_axis = util.check_slice_axis(slice_axis)
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = box_size // slice_size

    filenames = {
        0: f'x_slice_size_{util.num_to_str(slice_size)}.hdf5',
        1: f'y_slice_size_{util.num_to_str(slice_size)}.hdf5',
        2: f'z_slice_size_{util.num_to_str(slice_size)}.hdf5'
    }


    N_tot = sum(snap_info.num_part_tot)
    # crude estimate of maximum number of particles in each slice
    max_size = int(2 * N_tot / num_slices)

    # get the filename to save to
    if save_dir is None:
        save_dir = util.check_path(snap_info.filename.parent) / 'slices'
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
        for parttype in parttypes:
            h5file.create_dataset(
                f'{i}/PartType{parttype}/Coordinates', shape=(3, 0),
                dtype=float, maxshape=(3, max_size)
            )
            h5file.create_dataset(
                f'{i}/PartType{parttype}/Masses', shape=(0,),
                dtype=float, maxshape=(max_size,)
            )
            h5file.create_dataset(
                f'{i}/PartType{parttype}/IDs', shape=(0,),
                dtype=int, maxshape=(max_size,)
            )

    # now loop over all snapshot files and add their particle info
    # to the correct slice
    for num in tqdm(
            range(snap_info.num_files),
            desc='Slicing particle files'):
        # cycle through the different particle types
        for parttype in parttypes:
            # need to put particles along columns for hdf5 optimal usage
            coords = snap_info.read_single_file(
                i=num, var='PartType{parttype}/Coordinates',
                gadgetunits=False, verbose=False, reshape=True,
            ).T

            ids = snap_info.read_single_file(
                i=num, var='PartType{parttype}/ParticleIDs',
                gadgetunits=False, verbose=False, reshape=True,
            )

            # dark matter does not have the Mass variable
            if parttype != 1:
                masses = snap_info.read_single_file(
                    i=num, var='PartType{parttype}/Mass',
                    gadgetunits=False, verbose=False, reshape=True,
                )
            else:
                masses = np.atleast_1d(
                    snap_info.masses[parttype] * snap_info.mass_unit
                    / snap_info.solar_mass
                )

            slice_dict = ops.slice_particle_list(
                box_size=box_size,
                slice_size=slice_size,
                slice_axis=slice_axis,
                properties={
                    'coords': coords,
                    'ids': ids,
                    'masses': masses
                }
            )

            # append results to hdf5 file
            for idx, (coord, i) in enumerate(zip(
                    slice_dict['coords'],
                    slice_dict['ids'],
                    slice_dict['masses'])):
                if coord:
                    # add coordinates
                    dset_coords = h5file[f'{idx}/PartType{parttype}/Coordinates']
                    dset_coords.resize(
                        dset_coords.shape[-1] + coord[0].shape[-1], axis=1)
                    dset_coords[..., -coord[0].shape[-1]:] = coord[0]

                    # add masses
                    dset_masses = h5file[f'{idx}/PartType{parttype}/Masses']
                    dset_masses.resize(
                        dset_masses.shape[-1] + i[0].shape[-1], axis=0)
                    dset_masses[..., -i[0].shape[-1]:] = i[0]

                    # add particle IDs
                    dset_ids = h5file[f'{idx}/PartType{parttype}/IDs']
                    dset_ids.resize(
                        dset_ids.shape[-1] + i[0].shape[-1], axis=0)
                    dset_ids[..., -i[0].shape[-1]:] = i[0]

    h5file.close()
