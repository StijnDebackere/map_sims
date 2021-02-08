from gadget import Gadget
import h5py
import numpy as np
from tqdm import tqdm

import simulation_slices.io as io
import simulation_slices.hdf5_layouts as layouts
import simulation_slices.map_tools as map_tools
import simulation_slices.operations as ops
import simulation_slices.utilities as util

import pdb


def save_slice_data(
        base_dir, snapshot, datatype='snap', parttypes=[0, 1, 4, 5],
        slice_axis=0, slice_size=1, save_dir=None):
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
        model_dir=base_dir, file_type=datatype, snapnum=snapshot, sim='BAHAMAS',
        gadgetunits=True
    )

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(snap_info.filename.parent) / 'slices'
    else:
        save_dir = util.check_path(save_dir)
    util.make_dir(save_dir)

    # read snapshot properties
    box_size = snap_info.boxsize
    slice_axis = util.check_slice_axis(slice_axis)
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = int(box_size // slice_size)

    # crude estimate of maximum number of particles in each slice
    N_tot = sum(snap_info.num_part_tot)
    maxshape = int(2 * N_tot / num_slices)

    # set up to generate slice files
    filenames_base = {
        0: f'x_slice_size_{util.num_to_str(slice_size)}_{snapshot:03d}',
        1: f'y_slice_size_{util.num_to_str(slice_size)}_{snapshot:03d}',
        2: f'z_slice_size_{util.num_to_str(slice_size)}_{snapshot:03d}'
    }

    hdf_properties = layouts.bahamas_layout_properties(
        snap_info=snap_info, maxshape=maxshape
    )
    hdf_attrs = layouts.bahamas_attrs(
        snap_info=snap_info, slice_axis=slice_axis, slice_size=slice_size
    )

    # generate hd5file for each slice
    for i in range(num_slices):
        hdf_layout = layouts.get_slice_layout(
            slice_num=i, parttypes=parttypes, properties=hdf_properties,
            attrs=hdf_attrs,
        )

        filename = save_dir / f'{filenames_base[slice_axis]}_slice_num_{i}.hdf5'

        # ensure that we start with a clean slate
        print(f'Removing {filename.name} if it exists!')
        filename.unlink(missing_ok=True)

        # create hdf5 file and generate the required datasets
        io.create_hdf5(
            fname=filename, layout=hdf_layout, close=True
        )

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
                verbose=False, reshape=True,
            ).T

            # dark matter does not have the Mass variable
            if parttype != 1:
                # these masses are in solar masses, h has been filled in!
                masses = snap_info.read_single_file(
                    i=file_num, var=f'PartType{parttype}/Mass',
                    verbose=False, reshape=True,
                ) + 10
            else:
                # need to fill in h^-1 scaling
                masses = np.atleast_1d(snap_info.masses[parttype]) + 10

            properties = {
                    'coords': coords,
                    'masses': masses
            }
            # only gas particles have extra properties saved
            if parttype == 0:
                temperatures = snap_info.read_single_file(
                    i=file_num, var=f'PartType{parttype}/Temperature',
                    verbose=False, reshape=True,
                )
                densities = snap_info.read_single_file(
                    i=file_num, var=f'PartType{parttype}/Density',
                    verbose=False, reshape=True,
                )
                metallicities = snap_info.read_single_file(
                    i=file_num, var=f'PartType{parttype}/SmoothedMetallicity',
                    verbose=False, reshape=True,
                )
                properties = {
                    'temperatures': temperatures,
                    'densities': densities,
                    'metallicities': metallicities,
                    **properties
                }

            slice_dict = ops.slice_particle_list(
                box_size=box_size,
                slice_size=slice_size,
                slice_axis=slice_axis,
                properties=properties
            )

            # append results to hdf5 file
            for idx, (coord, masses) in enumerate(zip(
                    slice_dict['coords'],
                    slice_dict['masses'])):
                if coord:
                    fname = save_dir / f'{filenames_base[slice_axis]}_slice_num_{idx}.hdf5'
                    h5file = h5py.File(fname, 'r+')

                    # add coordinates
                    io.add_to_hdf5(
                        h5file=h5file, dataset=f'PartType{parttype}/Coordinates',
                        vals=coord[0], axis=1
                    )

                    # add masses
                    io.add_to_hdf5(
                        h5file=h5file, dataset=f'PartType{parttype}/Mass',
                        vals=masses[0], axis=0
                    )

                    if parttype == 0:
                        # get gas properties, list of array
                        temps = slice_dict['temperatures'][idx]
                        dens = slice_dict['densities'][idx]
                        metals = slice_dict['metallicities'][idx]

                        io.add_to_hdf5(
                            h5file=h5file, dataset=f'PartType{parttype}/Temperature',
                            vals=temps[0], axis=0
                        )
                        io.add_to_hdf5(
                            h5file=h5file, dataset=f'PartType{parttype}/Density',
                            vals=dens[0], axis=0
                        )
                        io.add_to_hdf5(
                            h5file=h5file, dataset=f'PartType{parttype}/SmoothedMetallicity',
                            vals=metals[0], axis=0
                        )


                    h5file.close()


def get_mass_projection_map(
        coord, slice_file, map_size, map_res, map_thickness, parttypes,
        verbose=True):
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
    verbose : bool
        show progress bar

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
    # create index array that cuts out the slice_axis
    no_slice_axis = range(coord.shape[0]) != slice_axis

    # add the projected mass map for each particle type to maps
    maps = []
    if verbose:
        iter_parttypes = tqdm(
            parttypes,
            desc=f'Joining slices for PartTypes')
    else:
        iter_parttypes = parttypes

    for parttype in iter_parttypes:
        coords = []
        masses = []
        for idx in range(slice_ids[0], slice_ids[1] + 1):
            # take into account periodic boundary conditions
            # all coordinates are 0 < x, y, z < L
            idx_mod = idx % num_slices
            coords.append(
                h5file[f'{idx_mod}/PartType{parttype}/Coordinates'][:]
            )
            masses.append(
                h5file[f'{idx_mod}/PartType{parttype}/Mass'][:]
            )

        coords = np.concatenate(coords, axis=-1)
        # now that we have all the particles that are roughly within
        # map_thickness, enforce map_thickness relative to coord
        selection = (
            (
                map_tools.dist(
                    coords[slice_axis].reshape(1, -1),
                    coord[slice_axis].reshape(1, -1), box_size, axis=0)
                <= map_thickness / 2
            ) & (
                map_tools.dist(
                    coords[no_slice_axis], coord[no_slice_axis].reshape(2, 1),
                    box_size, axis=0) <= 2**0.5 * map_size / 2
            )
        )

        # ignore sliced dimension
        coords_2d = coords[no_slice_axis][:, selection]
        map_center = coord[no_slice_axis]

        if parttype == 1:
            # all dark matter particles have the same mass
            masses = np.unique(np.concatenate(masses))
        else:
            masses = np.concatenate(masses)[selection]

        props = {'masses': masses}
        mp = ops.coords_to_map(
            coords=coords_2d, map_center=map_center, map_size=map_size,
            map_res=map_res, box_size=box_size, func=ops.masses,
            **props
        )

        maps.append(mp[None,...])

    h5file.close()
    maps = np.concatenate(maps, axis=0)
    return maps


def get_mass_projection_maps(coords, *args, **kwargs):
    """For all coords run get_mass_projection_map()."""
    maps = []
    for coord in coords:
        mp = get_mass_projection_map(coord, *args, **kwargs)
        maps.append(mp)

    maps = np.concatenate(maps, axis=0)
    return maps
