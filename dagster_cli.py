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
from numpy.random import SeedSequence, default_rng

import simulation_slices.tasks as tasks
from simulation_slices import Config

import dagster_solids as solids


parser = argparse.ArgumentParser(
    description="Run a dagster pipeline from config_filename."
)
parser.add_argument(
    "config_filename",
    default="/hpcdata0/simulations/BAHAMAS/extsdeba/batch_files/batch_bahamas_full.toml",
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
parser.add_argument("--save-coords", dest="save_coords", action="store_true")
parser.add_argument("--no-save-coords", dest="save_coords", action="store_false")
parser.set_defaults(save_coords=False)
parser.add_argument("--map-sims", dest="map_sims", action="store_true")
parser.add_argument("--no-map-sims", dest="map_sims", action="store_false")
parser.set_defaults(map_sims=True)


def pipeline_factory(config_filename: str, n_cpus: int):
    @pipeline(
        name="process_simulations",
        mode_defs=[
            ModeDefinition(
                executor_defs=[multiprocess_executor],
                resource_defs={
                    "io_manager": fs_io_manager,
                    "settings": make_values_resource(
                        save_coords=bool,
                        map_sims=bool,
                    ),
                },
            )
        ],
        description="Pipeline to generate observable maps from simulations.",
    )
    def pipeline_with_config():
        cfg = Config(config_filename)

        ss = SeedSequence(42)
        n_calls = cfg.snapshots.size * len(cfg.slice_axes)
        spawned = ss.spawn(n_calls)

        i = 0
        for sim_idx in range(cfg._n_sims):
            for idx_snap, snapshot in enumerate(cfg.snapshots[sim_idx]):
                save_coords = solids.save_coords_solid_factory(
                    sim_idx=sim_idx, snapshot=snapshot, cfg=cfg
                )
                coords = save_coords()

                for slice_axis in cfg.slice_axes:
                    rng = default_rng(spawned[i])
                    map_sim = solids.map_sim_solid_factory(
                        sim_idx=sim_idx,
                        snapshot=snapshot,
                        slice_axis=slice_axis,
                        rng=rng,
                        cfg=cfg,
                    )
                    mapped = map_sim(coords)
                    i += 1

    return pipeline_with_config


if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)

    os.environ["DAGSTER_HOME"] = dict_args["dagster_home"]
    reconstructable_pipeline = build_reconstructable_pipeline(
        "dagster_cli",
        "pipeline_factory",
        (),
        {
            "config_filename": dict_args["config_filename"],
            "n_cpus": dict_args["max_cpus"],
        },
    )
    result = execute_pipeline(
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
            "loggers": {"console": {"config": {"log_level": "INFO"}}},
            "resources": {
                "settings": {
                    "config": {
                        "save_coords": dict_args["save_coords"],
                        "map_sims": dict_args["map_sims"],
                    },
                },
            },
        },
    )
