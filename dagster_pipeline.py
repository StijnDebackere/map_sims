from pathlib import Path

from dagster import (
    ModeDefinition,
    execute_pipeline,
    multiprocess_executor,
    pipeline,
    solid,
)
from dagster.core.storage.fs_io_manager import fs_io_manager

import simulation_slices.batch_run as batch
from simulation_slices import Config


# inspired by https://stackoverflow.com/q/61330816/
def slice_sim_solid_factory(sim_idx, cfg):
    @solid(name=str(f"slice_sim_{sim_idx}"))
    def _slice_sim(context):
        context.log.info(f"Start slicing simulation {cfg.sim_dirs[sim_idx]}")
        batch.slice_sim(sim_idx=sim_idx, config=cfg)
        context.log.info(f"Finished slicing simulation {cfg.sim_dirs[sim_idx]}")
        return "Success"

    return _slice_sim


def map_sim_solid_factory(sim_idx, cfg):
    @solid(name=str(f"map_sim_{sim_idx}"))
    def _map_sim(context, result):
        context.log.info(f"Start mapping simulation {cfg.sim_dirs[sim_idx]}")
        batch.map_coords(sim_idx=sim_idx, config=cfg)
        context.log.info(f"Finished mapping simulation {cfg.sim_dirs[sim_idx]}")
        return "Success"

    return _map_sim


@solid
def summary_report(context, statuses):
    context.log.info(" ".join(statuses))


@pipeline(
    mode_defs=[
        ModeDefinition(
            executor_defs=[multiprocess_executor],
            resource_defs={"io_manager": fs_io_manager},
        )
    ]
)
def process_simulations():
    solid_output_handles = []
    cfg = Config(str(Path(__file__).parent / "simulation_slices/batch.toml"))

    for idx in range(cfg._n_sims):
        slice_sim = slice_sim_solid_factory(idx, cfg)
        map_sim = map_sim_solid_factory(idx, cfg)
        solid_output_handles.append(map_sim(slice_sim()))

    summary_report(solid_output_handles)


if __name__ == "__main__":
    execute_pipeline(
        pipeline,
        run_config={
            "execution": {"multiprocess_executor": {"config": {"max_concurrent": 2}}},
            "description": "Pipeline to generate observable maps from simulations.",
        },
    )
