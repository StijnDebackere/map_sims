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
import simulation_slices.maps.generation as map_gen
import simulation_slices.maps.interpolate_tables as interp_tables
import simulation_slices.maps.map_layout as map_layout
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
    downsample: bool = False,
    downsample_factor: float = None,
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
    if downsample:
        N_tot = downsample_factor * N_tot
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
            downsample=downsample,
            downsample_factor=downsample_factor,
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
        t0 = time.time()
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
                try:
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
                    elements_found = True
                    elements = {
                        "smoothed_hydrogen": smoothed_hydrogen,
                        "smoothed_helium": smoothed_helium,
                        "smoothed_carbon": smoothed_carbon,
                        "smoothed_nitrogen": smoothed_nitrogen,
                        "smoothed_oxygen": smoothed_oxygen,
                        "smoothed_neon": smoothed_neon,
                        "smoothed_magnesium": smoothed_magnesium,
                        "smoothed_silicon": smoothed_silicon,
                        "smoothed_iron": smoothed_iron,
                    }
                except KeyError:
                    elements_found = False
                    elements = {}


                # load in remaining data for X-ray luminosities
                properties = {
                    "temperatures": temperatures,
                    "densities": densities,
                    "masses": masses,
                    **elements,
                    **properties,
                }

            if downsample:
                n_part = properties["coordinates"].shape[-1]
                ids = np.random.choice(
                    n_part, replace=False, size=int(n_part * downsample_factor)
                )

                # rescale masses
                properties["masses"] = properties["masses"] / downsample_factor
                if properties["masses"].shape[0] == 1:
                    properties = dict(
                        (k, v[..., ids]) if k != "masses" else (k, v)
                        for k, v in properties.items()
                    )
                else:
                    properties = {k: v[..., ids] for k, v in properties.items()}

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
                    downsample=downsample,
                    downsample_factor=downsample_factor,
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
                        if elements_found:
                            for element in elements.keys():
                                element_vals = slice_dict[element][idx]
                                io.add_to_hdf5(
                                    h5file=h5file,
                                    vals=element_vals[0],
                                    axis=0,
                                    dataset=f'{idx}/{PROPS_PTYPES[ptype][element]}',
                                )

                h5file.close()
        t1 = time.time()
        if logger:
            logger.info(f"{file_num=} took {t1 - t0:.2f}s")

    return fnames


def save_maps_los(
    sim_dir: str,
    snapshot: int,
    centers: u.Quantity,
    group_ids: np.ndarray,
    masses: u.Quantity,
    slice_dir: str,
    slice_axis: int,
    num_slices: int,
    box_size: u.Quantity,
    map_pix: int,
    map_size: u.Quantity,
    map_thickness: u.Quantity,
    map_types: List[str],
    save_dir: str,
    coords_name: str = "",
    map_name_append: str = "",
    downsample: bool = False,
    downsample_factor: float = None,
    overwrite: bool = False,
    swmr: bool = False,
    method: str = None,
    n_ngb: int = 30,
    verbose: bool = False,
    logger: util.LoggerType = None,
) -> u.Quantity:
    """Project map around coord in a box of (map_size, map_size, map_thickness)
    in a map of (map_pix, map_pix) for mass of ptypes.

    Parameters
    ----------
    centers : (N, 3) astropy.units.Quantity
        (x, y, z) coordinates to slice around
    group_ids : (N,) np.ndarray
        group id for each coordinate
    masses : (N,) astropy.units.Quantity
        masses for each coordinate
    slice_dir : str
        directory of the saved simulation slices
    snapshot : int
        snapshot to look at
    slice_axis : int
        axis to slice along [x=0, y=1, z=2]
    box_size : astropy.units.Quantity
        size of simulation
    num_slices : int
        total number of slices
    map_pix : int
        square root of number of pixels in map
    map_size : astropy.units.Quantity
        size of the map
    map_thickness : astropy.units.Quantity
        thickness of the map projection
    map_types : ["gas_mass", "dm_mass", "stars_mass", "bh_mass"]
        particle types to compute masses for
    save_dir : str
        directory to save map files to
    coords_name : str
        identifier to append to filenames
    map_name_append : str
        optional extra to append to filenames
    overwrite : bool
        overwrite map_file if already exists
    swmr : bool
        enable single writer multiple reader mode for map_file
    method : str ["sph", "bin"]
        method for map projection: sph smoothing with n_ngb neighbours or 2D histogram
    n_ngb : int
        number of neighbours to determine SPH kernel size
    verbose : bool
        show progress bar

    Returns
    -------
    saves maps to {save_dir}/{slice_axis}_maps_{coords_name}{map_name_append}_{snapshot:03d}.hdf5
    """
    if not all(["mass" in map_type for map_type in map_types]):
        raise ValueError("only mass map_types are accepted")

    snap_info = Gadget(
        model_dir=sim_dir,
        file_type="snap",
        snapnum=snapshot,
        units=True,
        comoving=True,
    )

    map_thickness = map_thickness[np.argsort(map_thickness)[::-1]]
    map_name = map_gen.get_map_name(
        save_dir=save_dir,
        slice_axis=slice_axis,
        snapshot=snapshot,
        method=method,
        coords_name=coords_name,
        map_name_append=map_name_append,
        downsample=downsample,
        downsample_factor=downsample_factor,
        map_thickness=map_thickness,
    )

    # sort maps along mass
    centers = np.atleast_2d(centers).reshape(-1, 3)
    sort_ids = np.argsort(masses)[::-1]
    centers_sorted = centers[sort_ids]
    group_ids_sorted = group_ids[sort_ids]
    masses_sorted = masses[sort_ids]

    maxshape = centers.shape[0]

    # create index array that cuts out the slice_axis
    no_slice_axis = np.arange(0, 3) != slice_axis

    map_file = map_layout.create_map_file(
        map_name=map_name,
        overwrite=overwrite,
        close=False,
        swmr=swmr,
        slice_axis=slice_axis,
        box_size=box_size,
        map_types=map_types,
        map_size=map_size,
        map_thickness=map_thickness,
        map_pix=map_pix,
        snapshot=snapshot,
        n_ngb=n_ngb,
        maxshape=maxshape,
        extra={
            "centers": {
                "data": centers_sorted,
                "attrs": {
                    "description": "Halo centers.",
                    "single_value": False,
                    "units": str(centers_sorted.unit),
                },
            },
            "group_ids": {
                "data": group_ids_sorted,
                "attrs": {
                    "description": "Halo group ids.",
                    "single_value": False,
                },
            },
            "masses": {
                "data": masses_sorted,
                "attrs": {
                    "description": "Halo masses.",
                    "single_value": False,
                    "units": str(masses_sorted.unit),
                },
            },
        },
    )

    if not overwrite:
        min_idx = np.min([map_file[map_type].shape[0] for map_type in map_types])
        for map_type in map_types:
            # truncate all map_types to minimum size
            # might need to recalc some, but is easiest to implement
            map_file[map_type].resize(min_idx, axis=0)
    else:
        min_idx = 0

    # we will save maps in dictionary and write them to disk periodically
    pix_size = map_size / map_pix

    # only read in all coordinates once
    ptypes = [obs.MAP_TYPES_OPTIONS[map_type]["ptype"] for map_type in map_types]
    coords = {
        ptype: snap_info.read_var(PROPS_TO_BAHAMAS[ptype]["coordinates"])
        for ptype in set(ptypes)
    }
    maps = {
        map_type: [] for map_type in map_types
    }
    num_maps = 0
    for idx, (center, gid) in enumerate(zip(centers_sorted, group_ids_sorted)):
        t0 = time.time()
        num_maps += 1
        for ptype, map_type in zip(ptypes, map_types):
            # get rough boundary cuts for the map, allow some extra 2D space
            distance = np.ones(3) * 0.6 * map_size

            coords_slice = coords[ptype]
            # we will progressively shrink distance along slice_axis
            if ptype != "dm":
                masses_slice = snap_info.read_var(PROPS_TO_BAHAMAS[ptype]["masses"])
            else:
                masses_slice = np.atleast_1d(snap_info.masses[PTYPES_TO_BAHAMAS[ptype]])

            map_nslices = np.zeros((map_pix, map_pix, len(map_thickness))) * masses_slice.unit / map_size.unit ** 2
            # shrink map_thickness, shrinking coords and masses along the way
            for idx, dl in enumerate(map_thickness):
                ts = time.time()
                distance[slice_axis] = 0.5 * dl
                bounds = map_tools.slice_around_center(
                    center=center,
                    distance=distance,
                    box_size=box_size,
                    pix_size=None,
                )

                sl = np.zeros(coords_slice.shape, dtype=bool)
                for axis, bnds in bounds.items():
                    temp_sl = sl[:, axis]
                    # bounds possibly divided over opposite sides of periodic volume
                    for bnd in bnds:
                        if bnd[0].value == 0 * box_size.unit and bnd[1] == box_size:
                            temp_sl = np.ones_like(temp_sl)
                        else:
                            temp_sl = (
                                # need to be between bounds
                                (coords_slice[:, axis] >= bnd[0]) & (coords_slice[:, axis] <= bnd[1])
                                # and don't need to overlap with possible volume on other side of box
                                | temp_sl
                            )

                    sl[:, axis] = temp_sl

                # need to be inside bounds for all axes
                sl = np.all(sl, axis=-1)

                coords_slice = coords_slice[sl]
                if ptype != "dm":
                    masses_slice = masses_slice[sl]

                # ignore sliced dimension
                coords_2d = coords_slice[:, no_slice_axis]
                map_center = center[no_slice_axis]

                props_map_type = {"masses": masses_slice}
                # size of array required for SPH smoothing
                # SPH used up to haloes of 7152 particles for map_pix = 256
                # enough for the most massive downsampled MiraTitan haloes
                if method is None:
                    arr_size = 2 * coords_2d.shape[-1] * 2 * map_pix ** 2 * 64 * u.bit
                    if arr_size > 15 * u.GB:
                        coords_to_map = map_gen.coords_to_map_bin
                    else:
                        props_map_type = {"n_ngb": n_ngb, **props_map_type}
                        coords_to_map = map_gen.coords_to_map_sph

                elif method == "sph":
                    props_map_type = {"n_ngb": n_ngb, **props_map_type}
                    coords_to_map = map_gen.coords_to_map_sph
                elif method == "bin":
                    coords_to_map = map_gen.coords_to_map_bin
                else:
                    raise ValueError(f"{method=} should be 'sph' or 'bin'.")

                mp = coords_to_map(
                    # coords_to_map expect dimension along axis 0
                    coords=np.swapaxes(coords_2d, 0, 1),
                    map_center=map_center,
                    map_size=map_size,
                    map_pix=map_pix,
                    box_size=box_size,
                    func=obs.particles_masses,
                    logger=logger,
                    **props_map_type,
                )
                map_nslices[..., idx] = mp
                tf = time.time()
                if logger:
                    logger.info(
                        f"{gid=} - {map_type=} {dl=} took {tf - ts:.2f}s"
                    )

            # all map_thicknesses have been applied for map_type
            maps[map_type].append(map_nslices[None])
            if num_maps % 10 == 0:
                tw0 = time.time()
                # save after each slice_range
                io.add_to_hdf5(
                    h5file=map_file,
                    dataset=map_type,
                    vals=np.concatenate(maps[map_type], axis=0),
                    axis=0,
                )
                tw1 = time.time()
                if logger:
                    logger.info(
                        f"{gid=} - writing n={len(maps[map_type])} for {map_type=} took {tw1 - tw0:.2f}s"
                    )
                # start again from 0 maps
                maps[map_type] = []

    # write any remaining maps to file
    for map_type, mps in maps.items():
        if mps:
            io.add_to_hdf5(
                h5file=map_file,
                dataset=map_type,
                vals=np.concatenate(mps, axis=0),
                axis=0,
            )

    # still need to close the HDF5 files
    map_file.close()
    return map_name


def save_full_maps(
    sim_dir: str,
    snapshot: int,
    slice_axes: int,
    box_size: u.Quantity,
    map_types: List[str],
    map_pix: int,
    save_dir: str,
    map_name_append: str = "",
    downsample: bool = False,
    downsample_factor: float = None,
    overwrite: bool = False,
    swmr: bool = False,
    method: str = None,
    n_ngb: int = 30,
    verbose: bool = False,
    logger: util.LoggerType = None,
    **kwargs,
) -> List[str]:
    """Project full simulation in a map of (map_pix, map_pix) for slice_axes.

    Parameters
    ----------
    sim_dir : str
        directory of the simulation
    snapshot : int
        snapshot to look at
    slice_axes : int
        axis to slice along [x=0, y=1, z=2]
    box_size : astropy.units.Quantity
        size of simulation
    map_pix : int
        square root of number of pixels in map
    save_dir : str
        directory to save map files to
    map_name_append : str
        optional extra to append to filenames
    overwrite : bool
        overwrite map_file if already exists
    swmr : bool
        enable single writer multiple reader mode for map_file
    method : str ["sph", "bin"]
        method for map projection: sph smoothing with n_ngb neighbours or 2D histogram
    n_ngb : int
        number of neighbours to determine SPH kernel size
    verbose : bool
        show progress bar

    Returns
    -------
    saves maps to {save_dir}/{slice_axis}_maps_{coords_name}{map_name_append}_{snapshot:03d}.hdf5

    """
    t0 = time.time()
    slice_axes = np.atleast_1d(slice_axes)
    snap_info = Gadget(
        model_dir=sim_dir,
        file_type="snap",
        snapnum=snapshot,
        units=True,
        comoving=True,
    )

    # read in the Mpc unit box_size
    box_size = snap_info.boxsize
    h = snap_info.h

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(snap_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fnames = []
    map_files = {}
    for slice_axis in slice_axes:
        map_name = map_gen.get_map_name(
            save_dir=save_dir,
            slice_axis=slice_axis,
            snapshot=snapshot,
            method=method,
            map_thickness=box_size,
            coords_name="",
            map_name_append=map_name_append,
            downsample=downsample,
            downsample_factor=downsample_factor,
            full=True,
        )
        map_file = map_layout.create_map_file(
            map_name=map_name,
            overwrite=overwrite,
            close=False,
            # cannot have swmr since we are adding attributes later
            swmr=False,
            slice_axis=slice_axis,
            box_size=box_size,
            map_types=map_types,
            map_size=box_size,
            map_thickness=box_size,
            map_pix=map_pix,
            snapshot=snapshot,
            n_ngb=n_ngb,
            maxshape=0,
            full=True,
        )
        map_files[slice_axis] = {
            "map": {
                map_type: np.zeros((map_pix, map_pix), dtype=float)
                for map_type in map_types
            },
            "map_name": map_name,
            "map_file": map_file,
        }
        fnames.append(map_name)

    # now loop over all snapshot files and add their particle info
    # to the correct slice
    iterator = enumerate(range(snap_info.num_files))
    if verbose:
        iterator = tqdm(
            iterator,
            desc="Projecting particle files",
            total=snap_info.num_files,
        )

    for idx, file_num in iterator:
        for map_type in map_types:
            tl0 = time.time()
            ptype = obs.MAP_TYPES_OPTIONS[map_type]["ptype"]
            dsets = obs.MAP_TYPES_OPTIONS[map_type]["dsets"]
            attrs = obs.MAP_TYPES_OPTIONS[map_type].get("attrs", None)
            func  = obs.MAP_TYPES_OPTIONS[map_type]["func"]

            # only extract extra dsets
            dsets = list(set(dsets) - set(["coordinates", "masses"]))

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

            properties = {"masses": masses}
            if len(dsets) > 0:
                extra_props = {
                    prop: snap_info.read_single_file(
                        i=file_num,
                        var=PROPS_TO_BAHAMAS[ptype][prop],
                        verbose=False,
                        reshape=True,
                    ) for prop in dsets
                }
                properties = {**properties, **extra_props}

            if attrs is not None:
                for attr in attrs:
                    properties[attr] = getattr(snap_info, attr)

            tl1 = time.time()
            if logger:
                logger.info(
                    f"{file_num=} - loading properties took {tl1 - tl0:.2f}s"
                )

            # write each slice to a separate file
            for slice_axis in slice_axes:
                ts0 = time.time()
                no_slice_axis = np.arange(0, 3) != slice_axis
                if method == "bin":
                    coords_to_map = map_gen.coords_to_map_bin
                elif method == "sph":
                    coords_to_map = map_gen.coords_to_map_sph
                    properties = {**properties, "n_ngb": n_ngb}

                mp = coords_to_map(
                    coords=coords[no_slice_axis],
                    map_size=box_size,
                    map_pix=map_pix,
                    box_size=box_size,
                    func=func,
                    map_center=None,
                    logger=logger,
                    **properties
                )
                map_files[slice_axis]["map"][map_type] += mp.value
                ts1 = time.time()
                if logger:
                    logger.info(
                        f"{file_num=} - {slice_axis=} and {map_type=} took {ts1 - ts0:.2f}s"
                    )

                if idx == 0:
                    map_files[slice_axis]["map_file"][map_type].attrs["units"] = str(mp.unit)
                if idx % 10 == 0:
                    # save map to map_file and start again at 0
                    map_files[slice_axis]["map_file"][map_type][()] += map_files[slice_axis]["map"][map_type]
                    map_files[slice_axis]["map"][map_type] = np.zeros((map_pix, map_pix), dtype=float)
                    if logger:
                        logger.info(
                            f"{file_num=} - saved up to {idx=} for {slice_axis=} and {map_type=}"
                        )

    # append final remaining maps to map_file
    for slice_axis in slice_axes:
        for map_type in map_types:
            map_files[slice_axis]["map_file"][map_type][()] += map_files[slice_axis]["map"][map_type]

        map_files[slice_axis]["map_file"].close()

    t1 = time.time()
    if logger:
        logger.info(f"Finished {map_types=} and {slice_axes=} for {sim_dir=} took {t1 - t0:.2f}s")

    return fnames
