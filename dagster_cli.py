import argparse
import os
from pathlib import Path

from dagster import (
    DagsterInstance,
    execute_pipeline,
    pipeline,
    make_values_resource,
    ModeDefinition,
    multiprocess_executor,
)
from dagster.core.definitions.reconstructable import build_reconstructable_pipeline
from dagster.core.storage.fs_io_manager import fs_io_manager

import simulation_slices.tasks as tasks
from simulation_slices import Config

import dagster_solids as solids


parser = argparse.ArgumentParser(
    description="Run a dagster pipeline from config_filename."
)
parser.add_argument(
    "config_filename",
    default="/hpcdata0/simulations/BAHAMAS/extsdeba/batch_files/batch_bahamas_test.toml",
    type=str,
    help="configuration filename",
)
parser.add_argument(
    "-n",
    "--max_cpus",
    default=8,
    type=int,
    help="maximum number of concurrent processes",
)
parser.add_argument(
    "--dagster-home",
    default="/hpcdata0/simulations/BAHAMAS/extsdeba/dagster/",
    type=str,
    dest="dagster_home",
    help="$DAGSTER_HOME location",
)
parser.add_argument("--slice-sims", dest="slice_sims", action="store_true")
parser.add_argument("--no-slice-sims", dest="slice_sims", action="store_false")
parser.set_defaults(slice_sims=False)
parser.add_argument("--save-coords", dest="save_coords", action="store_true")
parser.add_argument("--no-save-coords", dest="save_coords", action="store_false")
parser.set_defaults(save_coords=False)
parser.add_argument("--map-sims", dest="map_sims", action="store_true")
parser.add_argument("--no-map-sims", dest="map_sims", action="store_false")
parser.set_defaults(map_sims=True)
parser.add_argument("--project-full", dest="project_full", action="store_true")
parser.add_argument("--no-project-full", dest="project_full", action="store_false")
parser.set_defaults(project_full=True)
parser.add_argument("--project-los", dest="project_los", action="store_true")
parser.add_argument("--no-project-los", dest="project_los", action="store_false")
parser.set_defaults(project_los=False)


def pipeline_factory(config_filename: str):
    @pipeline(
        name="process_simulations",
        mode_defs=[
            ModeDefinition(
                executor_defs=[multiprocess_executor],
                resource_defs={
                    "io_manager": fs_io_manager,
                    "settings": make_values_resource(
                        slice_sims=bool,
                        save_coords=bool,
                        map_sims=bool,
                        project_full=bool,
                        project_los=bool,
                    ),
                },
            )
        ],
        description="Pipeline to generate observable maps from simulations.",
    )
    def pipeline_with_config():
        solid_output_handles = []
        cfg = Config(config_filename)

        for sim_idx in range(cfg._n_sims):
            for idx_snap, snapshot in enumerate(cfg.snapshots[sim_idx]):
                slice_sim = solids.slice_sim_solid_factory(
                    sim_idx=sim_idx, snapshot=snapshot, cfg=cfg
                )
                save_coords = solids.save_coords_solid_factory(
                    sim_idx=sim_idx, snapshot=snapshot, cfg=cfg
                )

                coords_file = str(cfg.coords_files[sim_idx][idx_snap])
                map_sim = solids.map_sim_solid_factory(
                    sim_idx=sim_idx, snapshot=snapshot, coords_file=coords_file, cfg=cfg
                )

                solid_output_handles.append(map_sim(save_coords(slice_sim())))

    return pipeline_with_config


if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)

    os.environ["DAGSTER_HOME"] = dict_args["dagster_home"]
    reconstructable_pipeline = build_reconstructable_pipeline(
        "dagster_cli",
        "pipeline_factory",
        (),
        {"config_filename": dict_args["config_filename"]},
    )
    execute_pipeline(
        reconstructable_pipeline,
        instance=DagsterInstance.get(),
        run_config={
            "execution": {
                "multiprocess": {
                    "config": {
                        "max_concurrent": dict_args["max_cpus"],
                    }
                }
            },
            # "loggers": {"console": {"config": {"log_level": "INFO"}}},
            "resources": {
                "settings": {
                    "config": {
                        "slice_sims": dict_args["slice_sims"],
                        "save_coords": dict_args["save_coords"],
                        "map_sims": dict_args["map_sims"],
                        "project_full": dict_args["project_full"],
                        "project_los": dict_args["project_los"],
                    },
                },
            },
        },
    )
