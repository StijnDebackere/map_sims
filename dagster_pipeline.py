from pathlib import Path

from dagster import (
    execute_pipeline,
    pipeline,
    solid,
    make_values_resource,
    ModeDefinition,
    multiprocess_executor,
    Field,
    InputDefinition,
    Output,
    OutputDefinition,
    AssetMaterialization,
    AssetKey,
    EventMetadataEntry,
    Nothing,
)
from dagster.core.storage.fs_io_manager import fs_io_manager

import simulation_slices.tasks as tasks
from simulation_slices import Config


# inspired by https://stackoverflow.com/q/61330816/
def slice_sim_solid_factory(sim_idx: int, snapshot: int, cfg: Config):
    @solid(
        name=str(
            f"slice_sim_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{snapshot:03d}"
        ),
        description=f"Slice {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}.",
        required_resource_keys={"settings"},
    )
    def _slice_sim(context) -> Nothing:
        if context.resources.settings["slice_sims"]:
            context.log.info(
                f"Start slicing {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}"
            )
            if cfg.logging:
                logger = context.log
            else:
                logger = None

            fnames = tasks.slice_sim(
                sim_idx=sim_idx, snapshot=snapshot, config=cfg, logger=logger
            )
            context.log.info(
                f"Finished slicing {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}"
            )

            for idx, fname in enumerate(fnames):
                yield AssetMaterialization(
                    asset_key=AssetKey(
                        f"slice_sim_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{idx}_{snapshot:03d}"
                    ),
                    description=f"Slice file {idx} for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}",
                    metadata_entries=[EventMetadataEntry.path(fname, "file path")],
                )
        else:
            context.log.info(
                f"Skipping slicing {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )

        yield Output(None)

    return _slice_sim


def save_coords_solid_factory(sim_idx: int, snapshot: int, cfg: Config):
    @solid(
        name=str(
            f"save_coords_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{snapshot:03d}"
        ),
        input_defs=[InputDefinition("ready", dagster_type=Nothing)],
        description=f"Save coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}.",
        required_resource_keys={"settings"},
    )
    def _save_coords(context) -> Nothing:
        if context.resources.settings["save_coords"]:
            context.log.info(
                f"Start saving coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}"
            )
            if cfg.logging:
                logger = context.log
            else:
                logger = None

            fname = tasks.save_coords(
                sim_idx=sim_idx, snapshot=snapshot, config=cfg, logger=logger
            )

            context.log.info(
                f"Finished saving coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}"
            )

            yield AssetMaterialization(
                asset_key=AssetKey(
                    f"save_coords_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{snapshot:03d}"
                ),
                description=f"Coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}",
                metadata_entries=[EventMetadataEntry.path(fname, "file path")],
            )
        else:
            context.log.info(
                f"Skipping saving coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )

        yield Output(None)

    return _save_coords


def map_sim_solid_factory(sim_idx: int, snapshot: int, coords_file: str, cfg: Config):
    @solid(
        # needed output is list of AssetKeys from slice_sim
        input_defs=[InputDefinition("ready", dagster_type=Nothing)],
        name=str(
            f"map_sim_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{snapshot:03d}"
        ),
        description=f"Produce maps of {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}.",
        required_resource_keys={"settings"},
    )
    def _map_sim(context) -> Nothing:
        if context.resources.settings["map_sims"]:
            context.log.info(
                f"Start mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )
            if cfg.logging:
                logger = context.log
            else:
                logger = None

            fnames = []
            for slice_axis in cfg.slice_axes:
                if context.resources.settings["project_full"]:
                    fname = tasks.map_full(
                        sim_idx=sim_idx,
                        config=cfg,
                        snapshot=snapshot,
                        slice_axis=slice_axis,
                        logger=logger,
                    )
                    fnames.append(fname)
                else:
                    fname = tasks.map_coords(
                        sim_idx=sim_idx,
                        config=cfg,
                        snapshot=snapshot,
                        slice_axis=slice_axis,
                        coords_file=coords_file,
                        logger=logger,
                    )
                    fnames.append(fname)

            context.log.info(
                f"Finished mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )

            for idx, fname in enumerate(fnames):
                yield AssetMaterialization(
                    asset_key=f"map_sim_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{idx}_{snapshot:03d}",
                    description=f"Maps file {idx} for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}",
                    metadata_entries=[EventMetadataEntry.path(fname, "file path")],
                )
        else:
            context.log.info(
                f"Skipping mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )

        yield Output(None)

    return _map_sim


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
                ),
            },
        )
    ],
    description="Pipeline to generate observable maps from simulations.",
)
def pipeline():
    solid_output_handles = []
    cfg = Config("/hpcdata0/simulations/BAHAMAS/extsdeba/batch_files/batch_bahamas_full.toml")

    for sim_idx in range(cfg._n_sims):
        for idx_snap, snapshot in enumerate(cfg.snapshots[sim_idx]):
            slice_sim = slice_sim_solid_factory(
                sim_idx=sim_idx, snapshot=snapshot, cfg=cfg
            )
            save_coords = save_coords_solid_factory(
                sim_idx=sim_idx, snapshot=snapshot, cfg=cfg
            )

            coords_file = str(cfg.coords_files[sim_idx][idx_snap])
            map_sim = map_sim_solid_factory(
                sim_idx=sim_idx, snapshot=snapshot, coords_file=coords_file, cfg=cfg
            )

            solid_output_handles.append(map_sim(save_coords(slice_sim())))


if __name__ == "__main__":
    execute_pipeline(
        pipeline,
        run_config={
            "execution": {"multiprocess_executor": {"config": {"max_concurrent": 8}}},
            "loggers": {"console": {"config": {"log_level": "INFO"}}},
            "description": "Pipeline to generate observable maps from simulations.",
            "resources": {
                "settings": {
                    "config": {
                        "slice_sims": True,
                        "save_coords": True,
                        "map_sims": True,
                        "project_full": True,
                    },
                },
            },
        },
    )
