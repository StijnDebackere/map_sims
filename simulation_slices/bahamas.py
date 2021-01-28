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
        slice thickness in units of box_size
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
    num_slices = int(box_size // slice_size)

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
    else:
        save_dir = util.check_path(save_dir) / 'slices'

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
    h5file.attrs['a'] = snap_info.a
    h5file.attrs['h'] = snap_info.h

    for i in range(num_slices):
        for parttype in parttypes:
            # create coordinate datasets
            dset_coords = h5file.create_dataset(
                f'{i}/PartType{parttype}/Coordinates', shape=(3, 0),
                dtype=float, maxshape=(3, max_size)
            )
            dset_coords.attrs['CGSConversionFactor'] = snap_info.cm_per_mpc
            dset_coords.attrs['aexp-scale-exponent'] = 1.0
            dset_coords.attrs['h-scale-exponent'] = -1.0

            dset_mass = h5file.create_dataset(
                f'{i}/PartType{parttype}/Mass', shape=(0,),
                dtype=float, maxshape=(max_size,)
            )
            dset_mass.attrs['CGSConversionFactor'] = snap_info.mass_unit
            dset_mass.attrs['aexp-scale-exponent'] = 0.0
            dset_mass.attrs['h-scale-exponent'] = -1.0

            dset_ids = h5file.create_dataset(
                f'{i}/PartType{parttype}/ParticleIDs', shape=(0,),
                dtype=int, maxshape=(max_size,)
            )
            dset_ids.attrs['CGSConversionFactor'] = 1.0
            dset_ids.attrs['aexp-scale-exponent'] = 0.0
            dset_ids.attrs['h-scale-exponent'] = 0.0

    # now loop over all snapshot files and add their particle info
    # to the correct slice
    for file_num in tqdm(
            range(snap_info.num_files),
            desc='Slicing particle files'):
        # cycle through the different particle types
        for parttype in parttypes:
            # need to put particles along columns for hdf5 optimal usage
            # read everything in cMpc / h
            coords = snap_info.read_single_file(
                i=file_num, var=f'PartType{parttype}/Coordinates',
                gadgetunits=True, verbose=False, reshape=True,
            ).T

            ids = snap_info.read_single_file(
                i=file_num, var=f'PartType{parttype}/ParticleIDs',
                gadgetunits=True, verbose=False, reshape=True,
            )

            # dark matter does not have the Mass variable
            if parttype != 1:
                # these masses are in solar masses, h has been filled in!
                masses = snap_info.read_single_file(
                    i=file_num, var=f'PartType{parttype}/Mass',
                    gadgetunits=True, verbose=False, reshape=True,
                )
            else:
                # need to fill in h^-1 scaling
                masses = np.atleast_1d(snap_info.masses[parttype])

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
            for idx, (coord, i, masses) in enumerate(zip(
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
                    dset_masses = h5file[f'{idx}/PartType{parttype}/Mass']
                    dset_masses.resize(
                        dset_masses.shape[-1] + masses[0].shape[-1], axis=0)
                    dset_masses[..., -masses[0].shape[-1]:] = masses[0]

                    # add particle IDs
                    dset_ids = h5file[f'{idx}/PartType{parttype}/ParticleIDs']
                    dset_ids.resize(
                        dset_ids.shape[-1] + i[0].shape[-1], axis=0)
                    dset_ids[..., -i[0].shape[-1]:] = i[0]

    h5file.close()


@util.time_this
def get_mass_projection_map(
        coord, slice_file, map_size, map_res, map_thickness, parttypes):
    """Project mass around coord in a box of (map_size, map_size, slice_size)
    in a map of map_res.

    Parameters
    ----------
    coord : (3,) array
        (x, y, z) coordinates to slice around
    slice_file : str
        filename of the saved simulation slices
    map_size : float
        size of the map in units of box_size
    map_res : float
        resolution of the map in units of box_size
    map_thickness : float
        thickness of the map projection in units of box_size
    parttypes : [0, 1, 4, 5]
        particle types to include in projection

    Returns
    -------
    map : (map_size // map_res, map_size // map_res)
        pixelated projected mass

    """
    h5file = h5py.File(str(slice_file), mode='r')
    slice_axis = h5file.attrs['slice_axis']
    slice_size = h5file.attrs['slice_size']
    box_size = h5file.attrs['box_size']
    num_slices = int(box_size // slice_size)

    thickness = np.zeros((3,), dtype=float)
    thickness[slice_axis] += map_thickness / 2

    extent = np.array([
        coord - thickness, coord + thickness
    ]).T

    slice_ids = ops.get_coords_slices(
        coords=extent, slice_axis=slice_axis,
        slice_size=h5file.attrs['slice_size'],
    )

    # add the projected mass map for each particle type to maps
    maps = []
    for parttype in parttypes:
        coords = []
        masses = []
        for idx in range(*slice_ids):
            coords.append(h5file[f'{idx}/PartType{parttype}/Coordinates'][:])
            masses.append(h5file[f'{idx}/PartType{parttype}/Mass'][:])

        # ignore sliced dimension
        coords_2d = np.concatenate(coords)[range(coords.shape[0]) != slice_axis, :]
        map_center = coord[range(coord.shape[0]) != slice_axis]

        if parttype == 1:
            masses = np.unique(masses)
        else:
            masses = np.concatenate(masses)

        props = {'masses': masses}
        mp = ops.coords_to_map(
            coords=coords_2d, map_center=map_center, map_size=map_size,
            map_res=map_res, func_sum=ops.sum_masses, **props
        )

        maps.append(mp)

    h5file.close()
    return maps
