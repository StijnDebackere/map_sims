import argparse
import os
from pathlib import Path

import astropy.units as u
from tqdm import tqdm

import map_sims.io as io
import map_sims.tools as tools


parser = argparse.ArgumentParser(
    description="Join map_files matching base_fnames into fname."
)
parser.add_argument(
    "fname",
    default="BAHAMAS_aperture_masses_R2_2p5_Rm_5p0_clusters_m200m_gt_12p5.hdf5",
    type=str,
    help="filename to save joined hdf5 files to"
)
parser.add_argument(
    "-f", "--base_fnames",
    default="maps_bin_new_method_full_aperture_masses_R2_2p5_Rm_5p0_clusters_m200m_gt_12p5",
    type=str,
    nargs="+",
    help="looks for {slice_axis}_{base_fname}_{snapshot:03d}.hdf5"
)
parser.add_argument(
    "-d", "--dirs_file",
    default="join_mass_files_dirs.txt",
    type=str,
    help="file with all directories to search",
)
parser.add_argument(
    "-a", "--slice_axes",
    default=[0, 1, 2],
    type=int,
    nargs="+",
    help="slice_axes to look for",
)
parser.add_argument(
    "-s", "--snapshots",
    default=[22, 26, 28, 30],
    type=int,
    nargs="+",
    help="slice_axes to look for",
)
parser.add_argument(
    "--log_dir",
    default=".",
    type=str,
    help="base directory for simulations",
)


def main():
    args = parser.parse_args()

    dirs_file = args.dirs_file
    with open(dirs_file, "r") as f:
        dirs = [l.rstrip("\n") for l in f]

    save_fname = args.fname
    base_fnames = args.base_fnames
    log_dir = args.log_dir
    snapshots = args.snapshots
    slice_axes = args.slice_axes

    log_fname = f"{log_dir}/join_mass_files_{os.getpid()}"
    logger = tools.get_logger(fname=log_fname, log_level="info")


    fnames = [
        f"{slice_axis:d}_{base_fname}_{snapshot:03d}.hdf5"
        for slice_axis in slice_axes for base_fname in base_fnames for snapshot in snapshots
    ]
    final_dict = {}
    for idx, dr in enumerate(tqdm(dirs, desc="Loading simulations")):
        fnames_to_join = []
        for fname in fnames:
            file_path = f"{dr}/{fname}"
            if Path(file_path).exists():
                fnames_to_join.append(file_path)
            else:
                logger.info(f"{fname} not found in {dr}")

        for fname in fnames_to_join:
            final_dict = io.merge_dicts(
                a=final_dict,
                b=io.hdf5_to_dict(fname)
            )

        if idx % 10 == 0:
            io.dict_to_hdf5(
                fname=save_fname,
                data=final_dict,
                overwrite=True,
            )
            final_dict = {}

    if final_dict:
        io.dict_to_hdf5(
            fname=save_fname,
            data=final_dict,
            overwrite=True,
        )



if __name__ == "__main__":
    main()
