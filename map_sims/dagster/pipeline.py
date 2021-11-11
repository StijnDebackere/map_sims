from pathlib import Path

from dagster import (
    pipeline,
    make_values_resource,
    ModeDefinition,
    multiprocess_executor,
)
from dagster.core.storage.fs_io_manager import fs_io_manager

from map_sims import Config

import .solids as solids


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
def pipeline():
    cfg = Config("/hpcdata0/simulations/BAHAMAS/extsdeba/batch_files/batch_bahamas_full.toml")

    for sim_idx in range(cfg._n_sims):
        for idx_snap, snapshot in enumerate(cfg.snapshots[sim_idx]):
            save_info = solids.save_info_solid_factory(
                sim_idx=sim_idx, snapshot=snapshot, cfg=cfg
            )
            coords = save_coords()

            for slice_axis in cfg.slice_axes:
                map_sim = solids.map_sim_solid_factory(
                    sim_idx=sim_idx, snapshot=snapshot, slice_axis=slice_axis, cfg=cfg
                )
                mapped = map_sim(coords)
