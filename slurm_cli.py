import argparse
import os

import simulation_slices.tasks as tasks


parser = argparse.ArgumentParser(
    description="Run a dagster pipeline from config_filename."
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
    cfg = tasks.Config(dict_args["config_filename"])
    snapshot = dict_args["snapshot"]
    for sim_idx in range(cfg._n_sims):
        if dict_args["save_coords"]:
            results_coords = tasks.save_coords(
                sim_idx=sim_idx,
                snapshot=snapshot,
                config=cfg,
                logger=None,
            )
        if dict_args["project_full"]:
            results_full = tasks.map_full(
                sim_idx=sim_idx,
                config=cfg,
                snapshot=snapshot,
                logger=None,
            )
