import argparse
import os

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
parser.add_argument("--save-coords", dest="save_coords", action="store_true")
parser.add_argument("--no-save-coords", dest="save_coords", action="store_false")
parser.set_defaults(save_coords=False)


if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)

    # ensure that mira_titan does not claim all resources
    os.environ["OMP_NUM_THREADS"] = "1"
    snapshot = dict_args["snapshot"]
    sim_ids = dict_args["sim_ids"]

    cfg = tasks.Config(dict_args["config_filename"])
    # convert to strings instead of PosixPaths
    sims = [str(d) for d in cfg.sim_dirs]
    slice_axes = cfg.slice_axes

    print(f"Running {sims=} with {dict_args['save_coords']=} and {dict_args['project_full']=}")
    for sim_idx in sim_ids:
        if dict_args["save_coords"]:
            print(f"Saving coords for {sims[sim_idx]=} and {snapshot=} with {cfg.coords_name=}")
            results_coords = tasks.save_coords(
                sim_idx=sim_idx,
                snapshot=snapshot,
                config=cfg,
                logger=None,
            )
            print(f"Finished coords")
        if dict_args["project_full"]:
            print(f"Saving map_full for {sims[sim_idx]=} and {snapshot=} with {cfg.coords_name=}")
            for slice_axis in slice_axes:
                results_full = tasks.map_full(
                    sim_idx=sim_idx,
                    config=cfg,
                    snapshot=snapshot,
                    slice_axis=slice_axis,
                    logger=None,
                    rng=None,
                )
                print("Finished {slice_axis=}")
