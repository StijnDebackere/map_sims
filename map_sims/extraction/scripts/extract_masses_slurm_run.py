"""Command-line tool to start an sbatch run invoking extract_masses_cli.py

Usage: python extract_masses_slurm_run.py --maps-file /path/to/maps_file --info-file /path/to/info_file --radii-file /path/to/radii_file --n-tasks 8

maps_file: text file containing projected mass maps generated by map_sims.maps.generation.save_full_maps()
info_file: text file containing halo info generated by map_sims.sims.xxx.save_halo_info_file()
radii_file: toml file containing section 'radii' with keys r_aps, r_ins, r_out with matching shapes
"""
#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser(
    description="Submit maps-file to n-tasks slurm runs using extract_masses_sbatch.py"
)
parser.add_argument(
    "map_names_file",
    default="",
    type=str,
    help="file with all maps files filenames",
)
parser.add_argument(
    "info_names_file",
    default="",
    type=str,
    help="file with all info_file filenames",
)
parser.add_argument(
    "--n-tasks",
    default=8,
    type=int,
    help="number of tasks to split maps_file in",
    dest="n_tasks",
)
parser.add_argument(
    "--sim-suite",
    default="miratitan",
    type=str,
    dest="sim_suite",
    help="simulation suite to run for",
)
parser.add_argument(
    "--base-dir",
    default="/cosmo/scratch/projects/MiraTitanU/Grid/",
    type=str,
    dest="base_dir",
    help="path to base of saved simulations for simulation suite",
)
parser.add_argument(
    "--log-dir",
    default="/cosmo/scratch/debackere/logs/",
    type=str,
    dest="log_dir",
    help="path to log files to",
)
parser.add_argument(
    "--radii-file",
    default="/cosmo/scratch/debackere/batch_files/r_0p5-1p5_r2_0p5-2p0_rm_3p0.toml",
    type=str,
    dest="radii_file",
    help="toml file containing the section 'radii' with keys r_aps, r_ins and r_out",
)
parser.add_argument(
    "--env",
    default="simulation_slices",
    type=str,
    dest="env",
    help="conda environment to use",
)


def main():
    args = parser.parse_args()

    n_tasks = args.n_tasks
    info_file = args.info_names_file
    maps_file = args.map_names_file
    radii_file = args.radii_file
    sim_suite = args.sim_suite
    base_dir = args.base_dir
    log_dir = args.log_dir
    env = args.env

    path = Path(__file__).parent
    p = subprocess.Popen(
        [
            "sbatch",
            # need to pass these arguments first
            f"--ntasks={n_tasks}",
            "--partition=all",
            "--cpus-per-task=1",
            "--mem-per-cpu=8g",
            f"--output={log_dir}/batch-%j.out",
            f"--error={log_dir}/batch-%j.err",
            "--time=30-00:00:00",
            f"{str(path)}/extract_masses_sbatch.py",
            # ENSURE CORRECT ORDER!
            # TODO: figure out --variables support for sbatch
            maps_file,
            info_file,
            radii_file,
            sim_suite,
            base_dir,
            log_dir,
            env
        ],
    )


if __name__ == "__main__":
    main()