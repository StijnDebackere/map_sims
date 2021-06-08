import logging
import time
from typing import List, Optional, Tuple

import astropy.units as u
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
BAHAMAS_TO_PTYPES = {0: "gas", 1: "dm", 4: "stars", 5: "bh"}
PTYPES_TO_BAHAMAS = {"gas": 0, "dm": 1, "stars": 4, "bh": 5}

# conversion between expected properties and BAHAMAS hdf5 datasets
PROPS_TO_BAHAMAS = {
    ptype: {
        "coordinates": f"PartType{PTYPES_TO_BAHAMAS[ptype]}/Coordinates",
        "masses": f"PartType{PTYPES_TO_BAHAMAS[ptype]}/Mass",
    }
    for ptype in ["gas", "dm", "stars", "bh"]
}
PROPS_TO_BAHAMAS["gas"] = {
    "temperatures": f"PartType0/Temperature",
    "densities": f"PartType0/Density",
    "smoothed_hydrogen": f"PartType0/SmoothedElementAbundance/Hydrogen",
    "smoothed_helium": f"PartType0/SmoothedElementAbundance/Helium",
    "smoothed_carbon": f"PartType0/SmoothedElementAbundance/Carbon",
    "smoothed_nitrogen": f"PartType0/SmoothedElementAbundance/Nitrogen",
    "smoothed_oxygen": f"PartType0/SmoothedElementAbundance/Oxygen",
    "smoothed_neon": f"PartType0/SmoothedElementAbundance/Neon",
    "smoothed_magnesium": f"PartType0/SmoothedElementAbundance/Magnesium",
    "smoothed_silicon": f"PartType0/SmoothedElementAbundance/Silicon",
    "smoothed_iron": f"PartType0/SmoothedElementAbundance/Iron",
    **PROPS_TO_BAHAMAS["gas"],
}

# get correct properties from BAHAMAS ptypes
PROPS_PTYPES = {
    ptype: {"coordinates": f"{ptype}/coordinates", "masses": f"{ptype}/masses"}
    for ptype in ["gas", "dm", "stars", "bh"]
}
PROPS_PTYPES["gas"] = {
    "temperatures": f"gas/temperatures",
    "densities": f"gas/densities",
    "electron_number_densities": f"gas/electron_number_densities",
    "luminosities": f"gas/luminosities",
    "smoothed_hydrogen": f"gas/smoothed_hydrogen",
    "smoothed_helium": f"gas/smoothed_helium",
    "smoothed_carbon": f"gas/smoothed_carbon",
    "smoothed_nitrogen": f"gas/smoothed_nitrogen",
    "smoothed_oxygen": f"gas/smoothed_oxygen",
    "smoothed_neon": f"gas/smoothed_neon",
    "smoothed_magnesium": f"gas/smoothed_magnesium",
    "smoothed_silicon": f"gas/smoothed_silicon",
    "smoothed_iron": f"gas/smoothed_iron",
    **PROPS_PTYPES["gas"],
}


def save_coords_file(
    sim_dir: str,
    snapshot: int,
    coord_dset: str,
    mass_dset: str,
    mass_range: Tuple[u.Quantity, u.Quantity],
    coord_range: u.Quantity = None,
    extra_dsets: List[str] = None,
    save_dir: Optional[str] = None,
    coords_fname: Optional[str] = "",
    verbose: Optional[bool] = False,
    sample_haloes_bins: Optional[dict] = None,
    logger: util.LoggerType = None,
    **kwargs,
) -> str:
    """For snapshot of simulation in sim_dir, save the coord_dset for
    given group_dset and group_range.

    Parameters
    ----------
    sim_dir : str
        path of the MiraTitanU directory
    snapshot : int
        snapshot to look at
    coord_dset : str
        hdf5 dataset containing the coordinates
    mass_dset : str
        hdf5 FOF dataset to get masses from
    mass_range : (min, max) tuple
        minimum and maximum value for mass_dset
    coord_range : (3, 2) array
        range for coordinates to include
    extra_dsets : iterable
        extra datasets to save to the file
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/maps/
    coords_fname : str
        name for the coordinates file without extension

    Returns
    -------
    fname : str
        fname of saved coordinates

    """
    group_info = Gadget(
        model_dir=sim_dir, file_type="subh", snapnum=snapshot, units=True, comoving=True
    )
    h = group_info.h

    if "FOF" not in mass_dset:
        raise ValueError("group_dset should be a FOF property")
    if "FOF" not in coord_dset:
        raise ValueError("coord_dset should be a FOF property")
    if not np.all(["FOF" in extra_dset for extra_dset in extra_dsets]):
        raise ValueError("extra_dsets should be FOF properties")

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(group_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fname = (save_dir / f"{coords_fname}_{snapshot:03d}").with_suffix(".hdf5")

    coordinates = group_info.read_var(coord_dset, verbose=verbose)
    masses = group_info.read_var(mass_dset, verbose=verbose)
    group_ids = np.arange(len(masses))

    mass_range = mass_range.to(masses.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    mass_selection = (masses > np.min(mass_range)) & (masses < np.max(mass_range))

    # also select coordinate range
    if coord_range is not None:
        coord_range = coord_range.to(coordinates.unit, equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
        coord_selection = np.all(
            [
                (coordinates[:, i] > np.min(coord_range[i])) & (coordinates[:, i] < np.max(coord_range[i]))
                for i in range(coord_range.shape[0])
            ], axis=0
        )
        selection = mass_selection & coord_selection
    else:
        selection = mass_selection

    # subsample the halo sample
    if sample_haloes_bins is not None:
        mass_bin_edges = sample_haloes_bins["mass_bin_edges"]
        n_in_bin = sample_haloes_bins["n_in_bin"]

        # group halo indices by mass bins
        sampled_ids = util.groupby(np.arange(0, masses.shape[0]), masses, mass_bin_edges)

        selection = []
        for i, ids in sampled_ids.items():
            # get number of haloes to draw for bin
            n = n_in_bin[i]
            if n >= len(ids):
                selection.append(ids)
            else:
                selection.append(np.random.choice(ids, size=n, replace=False))
        selection = np.concatenate(selection)

    coordinates = coordinates[selection]
    masses = masses[selection]
    group_ids = group_ids[selection].astype(int)


    extra = {
        extra_dset: {
            "data": group_info.read_var(extra_dset, verbose=verbose).value[selection],
            "attrs": {
                "units": str(group_info.get_units(extra_dset, 0, verbose=verbose).unit),
                **group_info.read_attrs(
                    extra_dset, ids=0, verbose=verbose, dtype=object
                ),
            },
        }
        for extra_dset in (extra_dsets or {})
    }

    layout = {
        "attrs": {
            "description": "File with selected coordinates for maps.",
        },
        "dsets": {
            "coordinates": {
                "data": coordinates.to_value("Mpc / littleh"),
                "attrs": {
                    "description": "Coordinates in cMpc/h",
                    "units": "Mpc / littleh",
                    "mass_dset": mass_dset,
                    "mass_range": mass_range.value,
                    "mass_range_units": str(mass_range.unit),
                },
            },
            "group_ids": {
                "data": group_ids,
                "attrs": {
                    "description": "Group IDs starting at 0",
                },
            },
            "masses": {
                "data": masses.to_value("Msun / littleh"),
                "attrs": {
                    "description": "Masses in Msun / h",
                    "units": "Msun / littleh",
                    },
            },
            **extra,
        },
    }

    if coord_range is not None:
        layout["dsets"]["coordinates"]["attrs"]["coord_range"] = coord_range.value
        layout["dsets"]["coordinates"]["attrs"]["coord_range_units"] = str(coord_range.unit)

    io.create_hdf5(fname=fname, layout=layout, overwrite=True, close=True)
    return str(fname)


def save_slice_data(
    sim_dir: str,
    snapshot: int,
    slice_axes: List[int],
    num_slices: int,
    ptypes: List[str],
    save_dir: Optional[str] = None,
    verbose: Optional[bool] = False,
    logger: util.LoggerType = None,
) -> List[str]:
    """For snapshot of simulation in sim_dir, slice the particle data for
    all ptypes along the slice_axis. Slices are saved in the Snapshots
    directory by default.

    Parameters
    ----------
    sim_dir : str
        path of the MiraTitanU directory
    snapshot : int
        snapshot to look at
    slice_axes : list of int
        axes to slice along [x=0, y=1, z=2]
    num_slices : int
        number of slices to divide box in
    ptypes : iterable of ['gas', 'dm', 'bh', 'stars']
        particle types to read in
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/slices/
    verbose : bool
        print progress bar
    logger : logging.Logger
        optional logging

    Returns
    -------
    fnames : list of saved filenames
    saves particles for each slice in the snapshot_xxx/slices/
    directory

    """
    slice_axes = np.atleast_1d(slice_axes)
    snap_info = Gadget(
        model_dir=sim_dir,
        file_type="snap",
        snapnum=snapshot,
        units=True,
        comoving=True,
    )
    box_size = snap_info.boxsize

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(snap_info.filename).parent / "slices"
    else:
        save_dir = util.check_path(save_dir)

    # crude estimate of maximum number of particles in each slice
    N_tot = sum(snap_info.num_part_tot)
    maxshape = int(2 * N_tot / num_slices)

    a = snap_info.a
    z = 1 / a - 1
    h = snap_info.h

    fnames = []
    for slice_axis in slice_axes:
        # create the hdf5 file to fill up
        fname = slicing.create_slice_file(
            save_dir=save_dir,
            snapshot=snapshot,
            box_size=box_size,
            z=z,
            a=a,
            h=h,
            ptypes=ptypes,
            num_slices=num_slices,
            slice_axis=slice_axis,
            maxshape=maxshape,
        )
        fnames.append(fname)

    # now loop over all snapshot files and add their particle info
    # to the correct slice

    if verbose:
        num_files_range = tqdm(
            range(snap_info.num_files), desc="Slicing particle files"
        )
    else:
        num_files_range = range(snap_info.num_files)

    for file_num in num_files_range:
        for ptype in ptypes:
            # need to put particles along columns for hdf5 optimal usage
            # read everything in Mpc / h
            coords = snap_info.read_single_file(
                i=file_num,
                var=PROPS_TO_BAHAMAS[ptype]["coordinates"],
                verbose=False,
                reshape=True,
            ).T

            # dark matter does not have the Mass variable
            # read in M_sun / h
            if ptype != "dm":
                masses = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["masses"],
                    verbose=False,
                    reshape=True,
                )
            else:
                masses = np.atleast_1d(snap_info.masses[PTYPES_TO_BAHAMAS[ptype]])

            properties = {"coordinates": coords, "masses": masses}
            # only gas particles have extra properties saved
            if ptype == "gas":
                # load in particledata for SZ & X-ray
                temperatures = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["temperatures"],
                    verbose=False,
                    reshape=True,
                )
                densities = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["densities"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_hydrogen = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_hydrogen"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_helium = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_helium"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_carbon = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_carbon"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_nitrogen = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_nitrogen"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_oxygen = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_oxygen"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_neon = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_neon"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_magnesium = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_magnesium"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_silicon = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_silicon"],
                    verbose=False,
                    reshape=True,
                )
                smoothed_iron = snap_info.read_single_file(
                    i=file_num,
                    var=PROPS_TO_BAHAMAS[ptype]["smoothed_iron"],
                    verbose=False,
                    reshape=True,
                )
                # electron_number_densities = interp_tables.n_e(
                #     z=z,
                #     T=temperatures,
                #     rho=densities,
                #     h=h,
                #     X=smoothed_hydrogen,
                #     Y=smoothed_helium,
                # )
                # luminosities = interp_tables.x_ray_luminosity(
                #     z=z,
                #     rho=densities,
                #     T=temperatures,
                #     masses=masses,
                #     hydrogen_mf=smoothed_hydrogen,
                #     helium_mf=smoothed_helium,
                #     carbon_mf=smoothed_carbon,
                #     nitrogen_mf=smoothed_nitrogen,
                #     oxygen_mf=smoothed_oxygen,
                #     neon_mf=smoothed_neon,
                #     magnesium_mf=smoothed_magnesium,
                #     silicon_mf=smoothed_silicon,
                #     iron_mf=smoothed_iron,
                #     h=h,
                # )

                # load in remaining data for X-ray luminosities

                properties = {
                    "temperatures": temperatures,
                    "densities": densities,
                    "smoothed_hydrogen": smoothed_hydrogen,
                    "smoothed_helium": smoothed_helium,
                    "masses": masses,
                    "smoothed_hydrogen": smoothed_hydrogen,
                    "smoothed_helium": smoothed_helium,
                    "smoothed_carbon": smoothed_carbon,
                    "smoothed_nitrogen": smoothed_nitrogen,
                    "smoothed_oxygen": smoothed_oxygen,
                    "smoothed_neon": smoothed_neon,
                    "smoothed_magnesium": smoothed_magnesium,
                    "smoothed_silicon": smoothed_silicon,
                    "smoothed_iron": smoothed_iron,
                    # "electron_number_densities": electron_number_densities,
                    # "luminosities": luminosities,
                    **properties,
                }

            # write each slice to a separate file
            for slice_axis in slice_axes:
                slice_dict = slicing.slice_particle_list(
                    box_size=box_size,
                    num_slices=num_slices,
                    slice_axis=slice_axis,
                    properties=properties,
                )

                fname = slicing.slice_file_name(
                    save_dir=save_dir,
                    slice_axis=slice_axis,
                    num_slices=num_slices,
                    snapshot=snapshot,
                )
                h5file = h5py.File(fname, "r+")

                # append results to hdf5 file
                for idx, (coord, masses) in enumerate(
                    zip(slice_dict["coordinates"], slice_dict["masses"])
                ):
                    if not coord:
                        continue

                    coord_dset = f'{idx}/{PROPS_PTYPES[ptype]["coordinates"]}'
                    io.add_to_hdf5(
                        h5file=h5file,
                        dataset=coord_dset,
                        vals=coord[0],
                        axis=1,
                    )

                    # add masses
                    mass_dset = f'{idx}/{PROPS_PTYPES[ptype]["masses"]}'
                    if ptype == "dm":
                        # only want to add single value for dm mass
                        if h5file[mass_dset].shape[0] == 0:
                            io.add_to_hdf5(
                                h5file=h5file,
                                dataset=mass_dset,
                                vals=np.unique(masses[0]),
                                axis=0,
                            )
                    else:
                        io.add_to_hdf5(
                            h5file=h5file,
                            dataset=mass_dset,
                            vals=masses[0],
                            axis=0,
                        )

                    # add extra gas properties
                    if ptype == "gas":
                        # get gas properties, list of array
                        temperatures = slice_dict["temperatures"][idx]
                        densities = slice_dict["densities"][idx]
                        smoothed_hydrogen = slice_dict["smoothed_hydrogen"][idx]
                        smoothed_helium = slice_dict["smoothed_helium"][idx]
                        smoothed_carbon = slice_dict["smoothed_carbon"][idx]
                        smoothed_nitrogen = slice_dict["smoothed_nitrogen"][idx]
                        smoothed_oxygen = slice_dict["smoothed_oxygen"][idx]
                        smoothed_neon = slice_dict["smoothed_neon"][idx]
                        smoothed_magnesium = slice_dict["smoothed_magnesium"][idx]
                        smoothed_silicon = slice_dict["smoothed_silicon"][idx]
                        smoothed_iron = slice_dict["smoothed_iron"][idx]
                        # ne = slice_dict["electron_number_densities"][idx]
                        # lums = slice_dict["luminosities"][idx]

                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=temperatures[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["temperatures"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=densities[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["densities"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_hydrogen[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_hydrogen"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_helium[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_helium"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_carbon[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_carbon"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_nitrogen[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_nitrogen"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_oxygen[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_oxygen"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_neon[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_neon"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_magnesium[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_magnesium"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_silicon[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_silicon"]}',
                        )
                        io.add_to_hdf5(
                            h5file=h5file,
                            vals=smoothed_iron[0],
                            axis=0,
                            dataset=f'{idx}/{PROPS_PTYPES[ptype]["smoothed_iron"]}',
                        )
                        # io.add_to_hdf5(
                        #     h5file=h5file,
                        #     vals=ne[0],
                        #     axis=0,
                        #     dataset=f'{idx}/{PROPS_PTYPES[ptype]["electron_number_densities"]}',
                        # )
                        # io.add_to_hdf5(
                        #     h5file=h5file,
                        #     vals=lums[0],
                        #     axis=0,
                        #     dataset=f'{idx}/{PROPS_PTYPES[ptype]["luminosities"]}',
                        # )

                h5file.close()

    return fnames
