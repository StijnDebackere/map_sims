from gadget import Gadget
import h5py
import numpy as np
from tqdm import tqdm

import simulation_slices.io as io
import simulation_slices.sims.slicing as slicing
import simulation_slices.maps.tools as map_tools
import simulation_slices.maps.generation as gen
import simulation_slices.maps.interpolate_tables as interp_tables
import simulation_slices.maps.observables as obs
import simulation_slices.utilities as util


# conversion between BAHAMAS particle types and expected ones
BAHAMAS_TO_PTYPES = {
    0: 'gas',
    1: 'dm',
    4: 'stars',
    5: 'bh'
}

# conversion between expected properties and BAHAMAS hdf5 datasets
PROPS_TO_BAHAMAS = {
    i: {
        'coordinates': f'PartType{i}/Coordinates',
        'masses': f'PartType{i}/Mass'
    } for i in [0, 1, 4, 5]
}
PROPS_TO_BAHAMAS[0] = {
    'temperatures': f'PartType0/Temperature',
    'densities': f'PartType0/Density',
    'smoothed_hydrogen': f'PartType0/SmoothedElementAbundance/Hydrogen',
    'smoothed_helium': f'PartType0/SmoothedElementAbundance/Helium',
    'smoothed_carbon': f'PartType0/SmoothedElementAbundance/Carbon',
    'smoothed_nitrogen': f'PartType0/SmoothedElementAbundance/Nitrogen',
    'smoothed_oxygen': f'PartType0/SmoothedElementAbundance/Oxygen',
    'smoothed_neon': f'PartType0/SmoothedElementAbundance/Neon',
    'smoothed_magnesium': f'PartType0/SmoothedElementAbundance/Magnesium',
    'smoothed_silicon': f'PartType0/SmoothedElementAbundance/Silicon',
    'smoothed_iron': f'PartType0/SmoothedElementAbundance/Iron',
    **PROPS_TO_BAHAMAS[0]
}

# get correct properties from BAHAMAS ptypes
PROPS_PTYPES = {
    i: {
        'coordinates': f'{BAHAMAS_TO_PTYPES[i]}/coordinates',
        'masses': f'{BAHAMAS_TO_PTYPES[i]}/masses'
    } for i in [0, 1, 4, 5]
}
PROPS_PTYPES[0] = {
    'temperatures': f'gas/temperatures',
    'densities': f'gas/densities',
    'electron_number_densities': f'gas/electron_number_densities',
    'emissivities': f'gas/emissivities',
    'smoothed_hydrogen': f'gas/smoothed_hydrogen',
    'smoothed_helium': f'gas/smoothed_helium',
    'smoothed_carbon': f'gas/smoothed_carbon',
    'smoothed_nitrogen': f'gas/smoothed_nitrogen',
    'smoothed_oxygen': f'gas/smoothed_oxygen',
    'smoothed_neon': f'gas/smoothed_neon',
    'smoothed_magnesium': f'gas/smoothed_magnesium',
    'smoothed_silicon': f'gas/smoothed_silicon',
    'smoothed_iron': f'gas/smoothed_iron',
    **PROPS_PTYPES[0]
}

# enforce consistent units
# we want M_sun / h and cMpc / h
M_UNIT = 1e10
R_UNIT = 1

# and h^2 M_sun / Mpc^3
RHO_UNIT = M_UNIT / R_UNIT**3

DSET_UNITS = {
    'FOF/GroupMass': M_UNIT,
    'FOF/Group_M_TopHat200': M_UNIT,
    'FOF/Group_M_Mean200': M_UNIT,
    'FOF/Group_M_Crit200': M_UNIT,
    'FOF/Group_M_Mean500': M_UNIT,
    'FOF/Group_M_Crit500': M_UNIT,
    'FOF/Group_M_Mean2500': M_UNIT,
    'FOF/Group_M_Crit2500': M_UNIT,
    'FOF/Group_R_Mean200': R_UNIT,
    'FOF/Group_R_Crit200': R_UNIT,
    'FOF/Group_R_Mean500': R_UNIT,
    'FOF/Group_R_Crit500': R_UNIT,
    'FOF/Group_R_Mean2500': R_UNIT,
    'FOF/Group_R_Crit2500': R_UNIT,
    'FOF/GroupCentreOfPotential': R_UNIT,
}


def save_coords_file(
        base_dir, snapshot, group_dset, coord_dset, group_range, extra_dsets,
        save_dir=None, coords_fname='', verbose=False):
    """For snapshot of simulation in base_dir, save the coord_dset for
    given group_dset and group_range.

    Parameters
    ----------
    base_dir : str
        path of the MiraTitanU directory
    snapshot : int
        snapshot to look at
    group_dset : str
        hdf5 FOF dataset to select group from
    coord_dset : str
        hdf5 dataset containing the coordinates
    group_range : (min, max) tuple
        minimum and maximum value for group_dset
    extra_dsets : iterable
        extra datasets to save to the file
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/maps/
    coords_fname : str
        name for the coordinates file without extension

    Returns
    -------
    saves a set of coordinates to save_dir

    """
    group_info = Gadget(
        model_dir=base_dir, file_type='subh', snapnum=snapshot, sim='BAHAMAS',
        gadgetunits=True
    )

    if 'FOF' not in group_dset:
        raise ValueError('group_dset should be a FOF property')
    if 'FOF' not in coord_dset:
        raise ValueError('coord_dset should be a FOF property')
    if not np.all(['FOF' in extra_dset for extra_dset in extra_dsets]):
        raise ValueError('extra_dsets should be FOF properties')

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(group_info.filename).parent / 'maps'
    else:
        save_dir = util.check_path(save_dir)

    fname = (save_dir / coords_fname).with_suffix('.hdf5')

    group_data = group_info.read_var(group_dset, verbose=verbose) * DSET_UNITS.get(group_dset, 1.)
    group_ids = np.arange(len(group_data))
    selection = (group_data > np.min(group_range)) & (group_data < np.max(group_range))
    coordinates = group_info.read_var(coord_dset, verbose=verbose)[selection] * DSET_UNITS.get(coord_dset, 1.)

    extra = {
        extra_dset: {
            'data': group_info.read_var(extra_dset, verbose=verbose)[selection] * DSET_UNITS.get(extra_dset, 1.),
        }
        for extra_dset in extra_dsets
    }
    layout = {
        'attrs': {
            'description': 'File with selected coordinates for maps. All masses in M_sun/h',
        },
        'dsets': {
            'coordinates': {
                'data': coordinates,
                'attrs': {
                    'description': 'Coordinates in cMpc/h',
                    'group_dset': group_dset,
                    'group_range': group_range,

                },
            },
            'group_ids': {
                'data': group_ids[selection],
                'attrs': {
                    'description': 'Group IDs starting at 0',
                }
            },
            **extra,
        },
    }

    io.create_hdf5(
        fname=fname, layout=layout, close=True
    )


def save_slice_data(
        base_dir, snapshot, ptypes, slice_axes, slice_size,
        save_dir=None, verbose=False):
    """For snapshot of simulation in base_dir, slice the particle data for
    all ptypes along the x, y, and z directions. Slices are saved
    in the Snapshots directory by default.

    Parameters
    ----------
    base_dir : str
        path of the MiraTitanU directory
    datatype : str
        particles or snap, particle data to slice
    snapshot : int
        snapshot to look at
    ptypes : iterable of ints
        particle types to read in
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    slice_size : float
        slice thickness in units of box_size
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/slices/
    verbose : bool
        print progress bar

    Returns
    -------
    saves particles for each slice in the snapshot_xxx/slices/
    directory

    """
    slice_axes = np.atleast_1d(slice_axes)
    snap_info = Gadget(
        model_dir=base_dir, file_type='snap', snapnum=snapshot, sim='BAHAMAS',
        gadgetunits=True
    )

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(snap_info.filename).parent / 'slices'
    else:
        save_dir = util.check_path(save_dir)

    box_size = snap_info.boxsize
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = int(box_size // slice_size)

    # crude estimate of maximum number of particles in each slice
    N_tot = sum(snap_info.num_part_tot)
    maxshape = int(2 * N_tot / num_slices)

    for slice_axis in slice_axes:
        # create the hdf5 file to fill up
        slicing.create_slice_file(
            save_dir=save_dir, snapshot=snapshot, box_size=box_size,
            ptypes=[BAHAMAS_TO_PTYPES[p] for p in ptypes],
            num_slices=num_slices, slice_axis=slice_axis,
            slice_size=slice_size, maxshape=maxshape
        )

    # set unit conversions
    a = snap_info.a
    z = 1 / a - 1

    # now loop over all snapshot files and add their particle info
    # to the correct slice

    if verbose:
        num_files_range = tqdm(
            range(snap_info.num_files),
            desc='Slicing particle files'
        )
    else:
        num_files_range = range(snap_info.num_files)

    for file_num in num_files_range:
        for ptype in ptypes:
            # need to put particles along columns for hdf5 optimal usage
            # read everything in Mpc / h
            coords = snap_info.read_single_file(
                i=file_num, var=PROPS_TO_BAHAMAS[ptype]['coordinates'],
                verbose=False, reshape=True,
            ).T * R_UNIT

            # dark matter does not have the Mass variable
            # read in M_sun / h
            if ptype != 1:
                masses = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['masses'],
                    verbose=False, reshape=True,
                )
            else:
                masses = np.atleast_1d(snap_info.masses[ptype])

            masses *= M_UNIT

            properties = {
                'coordinates': coords,
                'masses': masses
            }
            # only gas particles have extra properties saved
            if ptype == 0:
                # load in particledata for SZ & X-ray
                temperatures = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['temperatures'],
                    verbose=False, reshape=True,
                )
                densities = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['densities'],
                    verbose=False, reshape=True,
                )
                densities *= RHO_UNIT

                smoothed_hydrogen = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_hydrogen'],
                    verbose=False, reshape=True,
                )
                smoothed_helium = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_helium'],
                    verbose=False, reshape=True,
                )
                smoothed_carbon = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_carbon'],
                    verbose=False, reshape=True,
                )
                smoothed_nitrogen = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_nitrogen'],
                    verbose=False, reshape=True,
                )
                smoothed_oxygen = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_oxygen'],
                    verbose=False, reshape=True,
                )
                smoothed_neon = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_neon'],
                    verbose=False, reshape=True,
                )
                smoothed_magnesium = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_magnesium'],
                    verbose=False, reshape=True,
                )
                smoothed_silicon = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_silicon'],
                    verbose=False, reshape=True,
                )
                smoothed_iron = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['smoothed_iron'],
                    verbose=False, reshape=True,
                )

                electron_number_densities = interp_tables.n_e(
                    z=z, T=temperatures, rho=densities,
                    X=smoothed_hydrogen, Y=smoothed_helium,
                )
                emissivities = interp_tables.x_ray_emissivity(
                    z=z, rho=densities, T=temperatures,
                    hydrogen_mf=smoothed_hydrogen, helium_mf=smoothed_helium,
                    carbon_mf=smoothed_carbon, nitrogen_mf=smoothed_nitrogen,
                    oxygen_mf=smoothed_oxygen, neon_mf=smoothed_neon,
                    magnesium_mf=smoothed_magnesium, silicon_mf=smoothed_silicon,
                    iron_mf=smoothed_iron,
                )

                # load in remaining data for X-ray luminosities

                properties = {
                    'temperatures': temperatures,
                    'densities': densities,
                    'electron_number_densities': electron_number_densities,
                    'emissivities': emissivities,
                    **properties
                }

            # write each slice to a separate file
            for slice_axis in slice_axes:
                slice_dict = slicing.slice_particle_list(
                    box_size=box_size,
                    slice_size=slice_size,
                    slice_axis=slice_axis,
                    properties=properties
                )

                fname = slicing.slice_file_name(
                    save_dir=save_dir, slice_axis=slice_axis,
                    slice_size=slice_size, snapshot=snapshot
                )
                h5file = h5py.File(fname, 'r+')

                # append results to hdf5 file
                for idx, (coord, masses) in enumerate(zip(
                        slice_dict['coordinates'],
                        slice_dict['masses'])):
                    if not coord:
                        continue

                    io.add_to_hdf5(
                        h5file=h5file, dataset=f'{idx}/{PROPS_PTYPES[ptype]["coordinates"]}',
                        vals=coord[0], axis=1
                    )

                    # add masses
                    if ptype == 1:
                        # only want to add single value for dm mass
                        if h5file[f'{idx}/{PROPS_PTYPES[ptype]["masses"]}'].shape[0] == 0:
                            io.add_to_hdf5(
                                h5file=h5file, dataset=f'{idx}/{PROPS_PTYPES[ptype]["masses"]}',
                                vals=np.unique(masses[0]), axis=0
                            )
                    else:
                        io.add_to_hdf5(
                            h5file=h5file, dataset=f'{idx}/{PROPS_PTYPES[ptype]["masses"]}',
                            vals=masses[0], axis=0
                        )

                    # add extra gas properties
                    if ptype == 0:
                        # get gas properties, list of array
                        temps = slice_dict['temperatures'][idx]
                        dens = slice_dict['densities'][idx]
                        ne = slice_dict['electron_number_densities'][idx]
                        ems = slice_dict['emissivities'][idx]
                        # hydrogen = slice_dict['smoothed_hydrogen'][idx]
                        # helium = slice_dict['smoothed_helium'][idx]

                        io.add_to_hdf5(
                            h5file=h5file, vals=temps[0], axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["temperatures"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file, vals=dens[0], axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["densities"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file, vals=ne[0], axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["electron_number_densities"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file, vals=ems[0], axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["emissivities"]}',
                        )

                h5file.close()
