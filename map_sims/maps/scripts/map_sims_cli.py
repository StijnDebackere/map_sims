#!/usr/bin/env python3
import argparse
import os

import numpy as np

import map_sims.tasks as tasks


parser = argparse.ArgumentParser(
    description="Run a simple pipeline for sim_ids and snapshot in config_filename."
)
parser.add_argument(
    "config_filename",
    default="",
    type=str,
    help="configuration filename",
)
parser.add_argument(
    "snapshot",
    default=401,
    type=int,
    help="snapshot to evaluate",
)
parser.add_argument(
    "-i", "--sim_ids",
    default=[0],
    type=int,
    nargs="+",
    help="index of config_file.sim_dirs to load",
)
parser.add_argument("--project-full", dest="project_full", action="store_true")
parser.add_argument("--no-project-full", dest="project_full", action="store_false")
parser.set_defaults(project_full=True)
parser.add_argument("--save-info", dest="save_info", action="store_true")
parser.add_argument("--no-save-info", dest="save_info", action="store_false")
parser.set_defaults(save_info=False)


def main():
    args = parser.parse_args()

    # ensure that mira_titan does not claim all resources
    os.environ["OMP_NUM_THREADS"] = "1"
    snapshot = args.snapshot
    sim_ids = args.sim_ids

    cfg = tasks.Config(args.config_filename)
    # convert to strings instead of PosixPaths
    sims = [str(d) for d in cfg.sim_dirs]
    slice_axes = cfg.slice_axes

    print(f"Running {sims=} with {args.save_info=} and {args.project_full=}")
    for sim_idx in sim_ids:
        if args.save_info:
            print(f"Saving info for {sims[sim_idx]=} and {snapshot=} with {cfg.info_name=}")
            results_info = tasks.save_info(
                sim_idx=sim_idx,
                snapshot=snapshot,
                config=cfg,
                logger=None,
            )
            print(f"Finished info")
        if args.project_full:
            print(f"Saving map_full for {sims[sim_idx]=} and {snapshot=} with {cfg.info_name=}")
            for slice_axis in slice_axes:
                results_full = tasks.map_full(
                    sim_idx=sim_idx,
                    config=cfg,
                    snapshot=snapshot,
                    slice_axis=slice_axis,
                    logger=None,
                    rng=np.random.default_rng(42),
                )
                print(f"Finished {slice_axis=}")


if __name__ == "__main__":
    main()
