from pathlib import Path

from dagster import (
    pipeline,
    make_values_resource,
    ModeDefinition,
    multiprocess_executor,
)
from dagster.core.storage.fs_io_manager import fs_io_manager

from simulation_slices import Config

import dagster_solids as solids


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
def pipeline():
    solid_output_handles = []
    cfg = Config("/hpcdata0/simulations/BAHAMAS/extsdeba/batch_files/batch_bahamas_los.toml")

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
