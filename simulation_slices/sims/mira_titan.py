from typing import List, Optional, Tuple
from pathlib import Path

import h5py
from mira_titan import MiraTitan
import numpy as np
from tqdm import tqdm

import simulation_slices.io as io
import simulation_slices.sims.slicing as slicing
import simulation_slices.utilities as util


PROPS_PTYPES = {"coordinates": f"dm/coordinates", "masses": f"dm/masses"}


def save_coords_file(
    sim_dir: str,
    box_size: u.Quantity,
    snapshot: int,
    mass_range: Tuple[u.Quantity, u.Quantity],
    save_dir: Optional[str] = None,
    coords_fname: Optional[str] = "",
    **kwargs,
) -> str:
    """For snapshot of simulation in sim_dir, save coordinates of haloes
    within group_range.

    Parameters
    ----------
    sim_dir : str
        path of the simulation
    box_size : astropy.units.Quantity
        size of simulation
    snapshot : int
        snapshot to look at
    mass_range : (min, max) tuple
        minimum and maximum value for masses
    save_dir : str or None
        location to save to, defaults to snapshot_xxx/maps/
    coords_fname : str
        name for the coordinates file without extension

    Returns
    -------
    fname : str
        filename of coords file
    saves a set of coordinates to save_dir

    """
    sim_info = MiraTitan(
        sim_dir=sim_dir,
        box_size=box_size,
        snapnum=snapshot,
    )
    h = sim_info.cosmo["h"]

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.filename).parent / "maps"
    else:
        save_dir = util.check_path(save_dir)

    fname = (save_dir / f"{coords_fname}_{snapshot:03d}").with_suffix(".hdf5")

    group_data = sim_info.read_properties(
        "fof",
        [
            "fof_halo_mass",
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "fof_halo_tag",
        ],
    )
    group_ids = group_data["fof_halo_tag"]
    masses = group_data["fof_halo_mass"]
    selection = (masses > np.min(mass_range)) & (masses < np.max(mass_range))

    # only included selection
    coordinates = (
        np.vstack(
            [
                group_data["fof_halo_center_x"],
                group_data["fof_halo_center_y"],
                group_data["fof_halo_center_z"],
            ]
        )
        .T[selection]
        .to("Mpc", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc)))
    )
    masses = masses[selection].to(
        "Msun", equivalencies=u.with_H0(100 * h * u.km / (u.s * u.Mpc))
    )
    group_ids = group_ids[selection].astype(int)

    layout = {
        "attrs": {
            "description": "File with selected coordinates for maps.",
        },
        "dsets": {
            "coordinates": {
                "data": coordinates,
                "attrs": {
                    "description": "Centers",
                    "units": "Mpc",
                    "mass_range": mass_range.value,
                    "mass_range_units": str(mass_range.unit),
                },
            },
            "group_ids": {
                "data": group_ids,
                "attrs": {
                    "description": "Group IDs",
                },
            },
            "masses": {
                "data": masses,
                "attrs": {
                    "description": "Masses",
                    "units": "Msun",
                },
            },
        },
    }

    io.create_hdf5(fname=fname, layout=layout, close=True)
    return str(fname)


def save_slice_data(
    sim_dir: str,
    box_size: int = 2100,
    snapshot: int = 499,
    slice_axes: List[int] = [0, 1, 2],
    slice_size: int = 20,
    save_dir: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> List[str]:
    """For snapshot of simulation in sim_dir, slice the particle data for
    all ptypes along the x, y, and z directions. Slices are saved
    in the Snapshots directory by default.

    Parameters
    ----------
    sim_dir : str
        path of simulation
    box_size : int
        size of simulation
    snapshot : int
        snapshot to look at
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
    fnames : list of str
    saves particles for each slice in the snapshot_xxx/slices/
    directory

    """
    slice_axes = np.atleast_1d(slice_axes)
    sim_info = MiraTitan(
        sim_dir=sim_dir,
        box_size=box_size,
        snapnum=snapshot,
    )

    # ensure that save_dir exists
    if save_dir is None:
        save_dir = util.check_path(sim_info.get_fname("snap")).parent / "slices"
    else:
        save_dir = util.check_path(save_dir)

    box_size = sim_info.box_size
    slice_size = util.check_slice_size(slice_size=slice_size, box_size=box_size)
    num_slices = int(box_size // slice_size)

    # crude estimate of maximum number of particles in each slice
    N_tot = sim_info.num_part_tot
    maxshape = int(2 * N_tot / num_slices)

    fnames = []
    for slice_axis in slice_axes:
        # create the hdf5 file to fill up
        fname = slicing.create_slice_file(
            save_dir=save_dir,
            snapshot=snapshot,
            box_size=sim_info.box_size.value,
            z=sim_info.z,
            a=sim_info.a,
            ptypes=["dm"],
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_size=slice_size.value,
            maxshape=maxshape,
        )
        fnames.append(fname)

    # now loop over all snapshot files and add their particle info
    # to the correct slice

    if verbose:
        num_files_range = tqdm(
            sim_info.datatype_info["snap"]["nums"], desc="Slicing particle files"
        )
    else:
        num_files_range = sim_info.datatype_info["snap"]["nums"]

    for file_num in num_files_range:
        properties = sim_info.read_properties(
            datatype="snap", props=["x", "y", "z"], num=file_num
        )
        coords = np.vstack([properties["x"], properties["y"], properties["z"]])
        masses = np.atleast_1d(sim_info.simulation_info["m_p"])

        properties = {"coordinates": coords, "masses": masses}
        # write each slice to a separate file
        for slice_axis in slice_axes:
            slice_dict = slicing.slice_particle_list(
                box_size=box_size,
                slice_size=slice_size,
                slice_axis=slice_axis,
                properties=properties,
            )

            fname = slicing.slice_file_name(
                save_dir=save_dir,
                slice_axis=slice_axis,
                slice_size=slice_size.value,
                snapshot=snapshot,
            )
            h5file = h5py.File(fname, "r+")

            # append results to hdf5 file
            for idx, (coord, masses) in enumerate(
                zip(slice_dict["coordinates"], slice_dict["masses"])
            ):
                if not coord:
                    continue

                io.add_to_hdf5(
                    h5file=h5file,
                    dataset=f'{idx}/{PROPS_PTYPES["coordinates"]}',
                    vals=coord[0],
                    axis=1,
                )

                # only want to add single value for dm mass
                if h5file[f'{idx}/{PROPS_PTYPES["masses"]}'].shape[0] == 0:
                    io.add_to_hdf5(
                        h5file=h5file,
                        dataset=f'{idx}/{PROPS_PTYPES["masses"]}',
                        vals=np.unique(masses[0]),
                        axis=0,
                    )

            h5file.close()

    return fnames
