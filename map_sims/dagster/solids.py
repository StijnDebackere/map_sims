from typing import Optional

from dagster import (
    solid,
    InputDefinition,
    Output,
    OutputDefinition,
    AssetMaterialization,
    AssetKey,
    EventMetadataEntry,
    Nothing,
)
from numpy.random import Generator
import map_sims.tasks as tasks
from map_sims import Config


# inspired by https://stackoverflow.com/q/61330816/
def save_info_solid_factory(sim_idx: int, snapshot: int, cfg: Config):
    @solid(
        name=str(
            f"save_info_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{snapshot:03d}"
        ),
        input_defs=[InputDefinition("ready", dagster_type=Nothing)],
        description=f"Save coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}.",
        required_resource_keys={"settings"},
    )
    def _save_info(context) -> Nothing:
        if context.resources.settings["save_info"]:
            context.log.info(
                f"Start saving coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}"
            )
            # dagster logger cannot log errors with exception info
            # run our own logger if cfg.logging
            logger = None

            fname = tasks.save_info(
                sim_idx=sim_idx, snapshot=snapshot, config=cfg, logger=logger
            )

            context.log.info(
                f"Finished saving coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}"
            )

            yield AssetMaterialization(
                asset_key=AssetKey(
                    f"save_info_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{snapshot:03d}"
                ),
                description=f"Coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} for {snapshot=}",
                metadata_entries=[EventMetadataEntry.path(fname, "file path")],
            )
        else:
            context.log.info(
                f"Skipping saving coordinates for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )

        yield Output(None)

    return _save_info


def map_sim_solid_factory(
    sim_idx: int,
    snapshot: int,
    slice_axis: int,
    cfg: Config,
    rng: Optional[Generator] = None,
):
    @solid(
        # needed output is list of AssetKeys from slice_sim
        input_defs=[InputDefinition("ready", dagster_type=Nothing)],
        name=str(
            f"map_sim_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            f"_snap_{snapshot:03d}_slice_{slice_axis}"
        ),
        description=str(
            f"Produce maps of {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} "
            f"for {snapshot=} and {slice_axis=}."
        ),
        required_resource_keys={"settings"},
    )
    def _map_sim(context) -> Nothing:
        if context.resources.settings["map_sims"]:
            context.log.info(
                f"Start mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} "
                f"{snapshot=:03d} and {slice_axis=}"
            )
            # dagster logger cannot log errors with exception info
            # run our own logger if cfg.logging
            logger = None

            fnames = tasks.map_full(
                sim_idx=sim_idx,
                snapshot=snapshot,
                slice_axis=slice_axis,
                config=cfg,
                logger=logger,
                rng=rng,
            )

            context.log.info(
                f"Finished mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} "
                f"{snapshot=:03d} and {slice_axis=}"
            )

            for idx, fname in enumerate(fnames):
                yield AssetMaterialization(
                    asset_key=f"map_sim_{str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}_{idx}_{snapshot:03d}",
                    description=f"Maps file {idx} for {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}",
                    metadata_entries=[EventMetadataEntry.path(fname, "file path")],
                )
        else:
            context.log.info(
                f"Skipping mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')} "
                f"{snapshot=:03d} and {slice_axis=}"
            )

        yield Output(None)

    return _map_sim
