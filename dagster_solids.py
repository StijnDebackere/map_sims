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
            # dagster logger cannot log errors with exception info
            # run our own logger if cfg.logging
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
            # dagster logger cannot log errors with exception info
            # run our own logger if cfg.logging
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
            # dagster logger cannot log errors with exception info
            # run our own logger if cfg.logging
            logger = None

            fnames_all = []
            if context.resources.settings["project_full"]:
                fnames = tasks.map_full(
                    sim_idx=sim_idx,
                    config=cfg,
                    snapshot=snapshot,
                    logger=logger,
                )
                fnames_all = [*fnames_all, *fnames]
            elif context.resources.settings["project_los"]:
                for slice_axis in cfg.slice_axes:
                    fname = tasks.map_los(
                        sim_idx=sim_idx,
                        snapshot=snapshot,
                        slice_axis=slice_axis,
                        coords_file=coords_file,
                        config=cfg,
                        logger=logger,
                    )
                    fnames_all.append(fname)
            else:
                for slice_axis in cfg.slice_axes:
                    fname = tasks.map_coords(
                        sim_idx=sim_idx,
                        config=cfg,
                        snapshot=snapshot,
                        slice_axis=slice_axis,
                        coords_file=coords_file,
                        logger=logger,
                    )
                    fnames_all.append(fname)

            context.log.info(
                f"Finished mapping simulation {str(cfg.sim_dirs[sim_idx]).replace('.', 'p')}"
            )

            for idx, fname in enumerate(fnames_all):
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
