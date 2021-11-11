import argparse
import itertools
import os

import astropy.units as u
import numpy as np
import map_sims.utilities as util
import map_sims.sims.bahamas as bahamas
import map_sims.sims.mira_titan as mira_titan


parser = argparse.ArgumentParser(
    description="Save coordinates for map_thickness range."
)
parser.add_argument(
    "base_save_dir",
    type=str,
    help="directory to save coordinate files to",
)
parser.add_argument(
    "info_base_fname",
    type=str,
    help="base filename of saved coordinates",
)
parser.add_argument(
    "sims_file",
    type=str,
    help="file with all simulation names",
)
parser.add_argument(
    "--sim_base_dir",
    default="/hpcdata0/simulations/BAHAMAS",
    type=str,
    help="base directory for simulations",
)
parser.add_argument(
    "--sim_suite",
    default="BAHAMAS",
    type=str,
    help="simulation suite",
)
parser.add_argument(
    "--snapshots",
    default=[28],
    type=int,
    nargs="+",
    help="snapshots to evaluate",
)
parser.add_argument(
    "--slice_axes",
    default=[0, 1, 2],
    type=int,
    nargs="+",
    help="slice_axes to evaluate",
)
parser.add_argument(
    "--L",
    default=400,
    type=float,
    help="box size",
)
parser.add_argument(
    "--m_min",
    default=12.5,
    type=float,
    help="log10 of minimum halo mass",
)
parser.add_argument(
    "--m_max",
    default=16.0,
    type=float,
    help="log10 of maximum halo mass",
)
parser.add_argument(
    "--l_frac",
    default=0.1,
    type=float,
    help="fraction of LOS depth from which to include haloes",
)
parser.add_argument(
    "--l_min",
    default=0.5,
    type=float,
    help="minimum LOS depth",
)
parser.add_argument(
    "--l_max",
    default=400,
    type=float,
    help="maximum LOS depth",
)
parser.add_argument(
    "--n_l",
    default=20,
    type=int,
    help="number of LOS slices",
)
parser.add_argument(
    "--log_dir",
    default="/hpcdata0/simulations/BAHAMAS/extsdeba/logs",
    type=str,
    help="location to save log file",
)
parser.add_argument("--overwrite", dest="overwrite", action="store_true")
parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
parser.set_defaults(overwrite=False)


def extract_fnames(file_path):
    with open(file_path, "r") as f:
        fnames = [l.rstrip("\n") for l in f]

    return fnames


def main():
    args = parser.parse_args()
    dict_args = vars(args)

    base_save_dir = dict_args["base_save_dir"]
    info_base_fname = dict_args["info_base_fname"]
    overwrite = dict_args["overwrite"]

    sim_suite = dict_args["sim_suite"].lower()
    sim_base_dir = dict_args["sim_base_dir"]
    sim_names = extract_fnames(dict_args["sims_file"])

    log_dir = dict_args["log_dir"]
    log_fname = f"batch_coords_{os.getpid()}"
    logger = util.get_logger(log_dir=log_dir, fname=log_fname, level="INFO")

    snapshots = dict_args["snapshots"]
    slice_axes = dict_args["slice_axes"]
    box_size = dict_args["L"]

    m_min = 10**dict_args["m_min"]
    m_max = 10**dict_args["m_max"]
    l_frac = dict_args["l_frac"]
    l_min = dict_args["l_min"]
    l_max = dict_args["l_max"]
    n_l = dict_args["n_l"]
    l_range = l_frac * np.geomspace(l_min, l_max, n_l)
    for sim_name, snapshot, slice_axis in itertools.product(
        sim_names, snapshots, slice_axes,
    ):
        sim_dir = f"{sim_base_dir}/{sim_name}/"
        save_dir = f"{base_save_dir}/{sim_name}/"
        for idx_l, dl in enumerate(l_range):
            info_fname = f"{slice_axis}_{info_base_fname}_slice_{idx_l}"

            if sim_suite.lower() == "bahamas":
                coords_range = np.array([[0, box_size]] * 3) * u.Mpc / u.littleh
                coords_range[slice_axis] = np.array(
                    [
                        0.5 * (box_size - dl), 0.5 * (box_size + dl)
                    ]
                ) * u.Mpc / u.littleh

                fn = bahamas.save_halo_info_file(
                    sim_dir=sim_dir,
                    snapshot=snapshot,
                    coord_dset="/FOF/GroupCentreOfPotential",
                    mass_dset="/FOF/Group_M_Mean200",
                    mass_range=np.array([m_min, m_max]) * u.Msun / u.littleh,
                    coord_range=coords_range,
                    extra_dsets=[
                        "/FOF/Group_M_Crit500",
                        "/FOF/Group_M_Crit200",
                    ],
                    save_dir=save_dir,
                    info_fname=info_fname,
                    verbose=False,
                    logger=logger,
                )

            elif sim_suite.lower() == "miratitan":
                coords_range = np.array([[0, box_size]] * 3)
                coords_range[:, slice_axis] = np.array(
                    [
                        0.5 * (box_size - dl), 0.5 * (box_size + dl)
                    ]
                )

                fn = mira_titan.save_halo_info_file(
                    sim_dir=sim_dir,
                    snapshot=snapshot,
                    mass_range=np.array([m_min, m_max]) * u.Msun,
                    coord_range=coords_range,
                    save_dir=save_dir,
                    info_fname=info_fname,
                    logger=logger,
                    verbose=False,
                )


if __name__ == "__main__":
    main()
