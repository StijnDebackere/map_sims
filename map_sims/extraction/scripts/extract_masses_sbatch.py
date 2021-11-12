#!/usr/bin/env python3


import argparse
import os
from pathlib import Path
import time
import subprocess
import sys


FILE_PATH = str(Path(__file__).parent)
NTASKS = int(os.environ["SLURM_NTASKS"])
NCPUS = int(os.environ["SLURM_CPUS_PER_TASK"])


# arguments passes by extract_masses_slurm_run.sh
parser = argparse.ArgumentParser(
    description="Split map_names_file into SLURM_NTASKS and pass off to extract_masses.cli"
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
    "sim_suite",
    default="miratitan",
    type=str,
    help="simulation suite to run for",
)
parser.add_argument(
    "base_dir",
    default="/cosmo/scratch/projects/MiraTitanU/Grid/",
    type=str,
    help="path to base of saved simulations for simulation suite",
)
parser.add_argument(
    "log_dir",
    default="/cosmo/scratch/projects/MiraTitanU/Grid/",
    type=str,
    help="path to log files to",
)
parser.add_argument(
    "env",
    default="simulation_slices",
    type=str,
    help="conda environment to use",
)


def extract_from_file(file_path):
    with open(file_path, "r") as f:
        fnames = [l.rstrip("\n") for l in f]

    return fnames


def write_to_file(file_path, lst):
    with open(file_path, "w") as f:
        for l in lst:
            f.write(f"{l}\n")


def main():
    args = parser.parse_args()

    info_file = args.info_names_file
    map_files = extract_from_file(args.map_names_file)
    sim_suite = args.sim_suite
    base_dir = args.base_dir
    log_dir = args.log_dir
    env = args.env

    # need to activate the correct conda environment for each srun call
    # see https://github.com/conda/conda/issues/9296#issuecomment-537085104
    CONDA_BASE = Path(os.environ["CONDA_EXE"]).parent.parent
    activate_env = f". {CONDA_BASE}/etc/profile.d/conda.sh && conda activate {env}"

    # divide map_files into NTASKS chunks
    num_per_chunk = int((len(map_files) + NTASKS - 1) / NTASKS)
    chunk_ids = range(0, len(map_files) + 1, num_per_chunk)
    if chunk_ids[-1] > len(map_files):
        chunk_ids[-1] = len(map_files)

    map_files_split = [
        map_files[i:j] for i, j in zip(chunk_ids[:-1], chunk_ids[1:])
    ]
    temp_map_filenames = [
        args.map_names_file + f".{i}.temp" for i in range(len(map_files_split))
    ]

    procs = []
    for temp_map_filename, maps in zip(temp_map_filenames, map_files_split):
        write_to_file(temp_map_filename, maps)
        srun_cmd = [
            "srun",
            "--ntasks=1",
            f"--cpus-per-task={NCPUS}",
            "--exclusive",
            "python",
            "{FILE_PATH}/extract_masses_cli.py",
            temp_map_filename,
            info_file,
            f"--sim_suite={SIM_SUITE}"
            f"--log_dir={LOG_DIR}",
            f"--base_dir={BASE_DIR}",
            "&",
        ]
        full_cmd = f"{activate_env} && {' '.join(srun_cmd)}"

        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            executable="/usr/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Submitted {temp_map_filename} to pid {proc.pid}")
        procs.append(proc)

    # check if all processes have finished
    while all([p.poll() is None for p in procs]):
        time.sleep(30)

    # check return codes
    return_codes = [p.poll() for p in procs]
    for idx, rc in enumerate(return_codes):
        print(
            f"{temp_map_filenames[idx]} output = ",
            procs[idx].stdout.read().decode("utf8"),
        )
        if rc != 0:
            print(f"{temp_map_filenames[idx]} exited with return code {rc}")
            print(procs[idx].stderr.read().decode("utf8"))

    # remove temp files
    for temp_map_filename in temp_map_filenames:
        print(f"Removing {temp_map_filename}")
        os.remove(temp_map_filename)


if __name__ == "__main__":
    main()
