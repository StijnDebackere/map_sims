#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess

import numpy as np

from simulation_slices import Config


parser = argparse.ArgumentParser(
    description="Submit multiple sbatch runs using sbatch_single.sh"
)
parser.add_argument(
    "--config",
    default="",
    type=str,
    help="configuration filename",
)
parser.add_argument(
    "--n-tasks",
    default=1,
    type=int,
    help="number of sbatch scripts to submit",
    dest="n_tasks"
)



def main():
    args = vars(parser.parse_args())
    n_tasks = args["n_tasks"]

    cfg_fname = args["config"]
    cfg = Config(cfg_fname)
    snapshots = cfg.snapshots
    if not np.all(snapshots == snapshots[0]):
        raise ValueError("can only pass all same snapshots to each sim in batch")

    snapshots = snapshots[0].astype(str).tolist()
    sims = [str(sim) for sim in cfg.sim_dirs]

    n_sims = len(cfg.sim_dirs)
    n_ids_per_task = int((n_sims + n_tasks - 1) / n_tasks)

    for i in range(0, n_tasks):
        n_start = i * n_ids_per_task
        if i == n_tasks - 1:
            n_stop = n_sims
        else:
            n_stop = (i + 1) * n_ids_per_task
        # sbatch_single argument INCLUDES final idx
        subprocess.run([
            "sbatch", "sbatch_single.sh", cfg_fname, str(n_start), str(n_stop - 1),
            f"--array={','.join(snapshots)}", f"--ntasks={len(snapshots)}"
        ])

        print(f"Submitted sbatch run for {cfg_fname=} and sims={','.join(sims[n_start:n_stop])} and snapshots={','.join(snapshots)}")



if __name__ == "__main__":
    main()