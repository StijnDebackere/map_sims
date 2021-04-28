from typing import List, Optional, Tuple

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
PTYPES_TO_BAHAMAS = {
    'gas': 0,
    'dm': 1,
    'stars': 4,
    'bh': 5
}

# conversion between expected properties and BAHAMAS hdf5 datasets
PROPS_TO_BAHAMAS = {
    ptype: {
        'coordinates': f'PartType{PTYPES_TO_BAHAMAS[ptype]}/Coordinates',
        'masses': f'PartType{PTYPES_TO_BAHAMAS[ptype]}/Mass'
    } for ptype in ['gas', 'dm', 'stars', 'bh']
}
PROPS_TO_BAHAMAS['gas'] = {
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
    **PROPS_TO_BAHAMAS['gas']
}

# get correct properties from BAHAMAS ptypes
PROPS_PTYPES = {
    ptype: {
        'coordinates': f'{ptype}/coordinates',
        'masses': f'{ptype}/masses'
    } for ptype in ['gas', 'dm', 'stars', 'bh']
}
PROPS_PTYPES['gas'] = {
    'temperatures': f'gas/temperatures',
    'densities': f'gas/densities',
    'electron_number_densities': f'gas/electron_number_densities',
    'luminosities': f'gas/luminosities',
    'smoothed_hydrogen': f'gas/smoothed_hydrogen',
    'smoothed_helium': f'gas/smoothed_helium',
    'smoothed_carbon': f'gas/smoothed_carbon',
    'smoothed_nitrogen': f'gas/smoothed_nitrogen',
    'smoothed_oxygen': f'gas/smoothed_oxygen',
    'smoothed_neon': f'gas/smoothed_neon',
    'smoothed_magnesium': f'gas/smoothed_magnesium',
    'smoothed_silicon': f'gas/smoothed_silicon',
    'smoothed_iron': f'gas/smoothed_iron',
    **PROPS_PTYPES['gas']
}


def save_coords_file(
        sim_dir: str, snapshot: int,
        group_dset: str, coord_dset: str,
        group_range: Tuple[float, float],
        extra_dsets: List[str],
        save_dir: Optional[str]=None,
        coords_fname: Optional[str]='',
        verbose: Optional[bool]=False) -> None:
    """For snapshot of simulation in sim_dir, save the coord_dset for
    given group_dset and group_range.

    Parameters
    ----------
    sim_dir : str
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
        model_dir=sim_dir, file_type='subh', snapnum=snapshot, sim='BAHAMAS',
        units=True, comoving=True
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

    group_data = group_info.read_var(group_dset, verbose=verbose)
    group_ids = np.arange(len(group_data))
    selection = (group_data > np.min(group_range)) & (group_data < np.max(group_range))
    coordinates = group_info.read_var(coord_dset, verbose=verbose)[selection]

    extra = {
        extra_dset: {
            'data': group_info.read_var(extra_dset, verbose=verbose).value[selection],
            'attrs': {
                'units': str(group_info.get_units(extra_dset, 0, verbose=verbose).unit),
                **group_info.read_attrs(extra_dset, ids=0, dtype=object)
            },
        }
        for extra_dset in extra_dsets
    }
    layout = {
        'attrs': {
            'description': 'File with selected coordinates for maps. All masses in M_sun/h',
        },
        'dsets': {
            'coordinates': {
                'data': coordinates.value,
                'attrs': {
                    'description': 'Coordinates in cMpc/h',
                    'units': str(coordinates.unit),
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
        sim_dir: str, snapshot: int, ptypes: List[str],
        slice_axes: List[int], slice_size: int,
        save_dir: Optional[str]=None,
        verbose: Optional[bool]=False) -> None:
    """For snapshot of simulation in sim_dir, slice the particle data for
    all ptypes along the x, y, and z directions. Slices are saved
    in the Snapshots directory by default.

    Parameters
    ----------
    sim_dir : str
        path of the MiraTitanU directory
    datatype : str
        particles or snap, particle data to slice
    snapshot : int
        snapshot to look at
    ptypes : iterable of ['gas', 'dm', 'bh', 'stars']
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
        model_dir=sim_dir, file_type='snap', snapnum=snapshot,
        units=True, comoving=True,
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

    a = snap_info.a
    z = 1 / a - 1
    h = snap_info.h

    for slice_axis in slice_axes:
        # create the hdf5 file to fill up
        slicing.create_slice_file(
            save_dir=save_dir, snapshot=snapshot, box_size=box_size.value,
            z=z, a=a, ptypes=ptypes,
            num_slices=num_slices, slice_axis=slice_axis,
            slice_size=slice_size.value, maxshape=maxshape
        )


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
            ).T

            # dark matter does not have the Mass variable
            # read in M_sun / h
            if ptype != 'dm':
                masses = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['masses'],
                    verbose=False, reshape=True,
                )
            else:
                masses = np.atleast_1d(snap_info.masses[PTYPES_TO_BAHAMAS[ptype]])

            properties = {
                'coordinates': coords,
                'masses': masses
            }
            # only gas particles have extra properties saved
            if ptype == 'gas':
                # load in particledata for SZ & X-ray
                temperatures = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['temperatures'],
                    verbose=False, reshape=True,
                )
                densities = snap_info.read_single_file(
                    i=file_num, var=PROPS_TO_BAHAMAS[ptype]['densities'],
                    verbose=False, reshape=True,
                )
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
                    z=z, T=temperatures, rho=densities, h=h,
                    X=smoothed_hydrogen, Y=smoothed_helium,
                )
                luminosities = interp_tables.x_ray_luminosity(
                    z=z, rho=densities, T=temperatures, masses=masses,
                    hydrogen_mf=smoothed_hydrogen, helium_mf=smoothed_helium,
                    carbon_mf=smoothed_carbon, nitrogen_mf=smoothed_nitrogen,
                    oxygen_mf=smoothed_oxygen, neon_mf=smoothed_neon,
                    magnesium_mf=smoothed_magnesium, silicon_mf=smoothed_silicon,
                    iron_mf=smoothed_iron, h=h
                )

                # load in remaining data for X-ray luminosities

                properties = {
                    'temperatures': temperatures,
                    'densities': densities,
                    'electron_number_densities': electron_number_densities,
                    'luminosities': luminosities,
                    **properties
                }

            # write each slice to a separate file
            for slice_axis in slice_axes:
                slice_dict = slicing.slice_particle_list(
                    box_size=box_size,
                    slice_size=slice_size,
                    slice_axis=slice_axis,
                    properties=properties,
                )

                fname = slicing.slice_file_name(
                    save_dir=save_dir, slice_axis=slice_axis,
                    slice_size=slice_size.value, snapshot=snapshot
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
                    if ptype == 'dm':
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
                    if ptype == 'gas':
                        # get gas properties, list of array
                        temps = slice_dict['temperatures'][idx]
                        dens = slice_dict['densities'][idx]
                        ne = slice_dict['electron_number_densities'][idx]
                        lums = slice_dict['luminosities'][idx]
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
                            h5file=h5file, vals=lums[0], axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["luminosities"]}',
                        )

                h5file.close()
