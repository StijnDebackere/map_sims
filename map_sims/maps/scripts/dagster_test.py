import os

from dagster import (
    DagsterInstance,
    execute_pipeline,
    ModeDefinition,
    multiprocess_executor,
    Nothing,
    Output,
    pipeline,
    solid,
)
from dagster.core.definitions.reconstructable import build_reconstructable_pipeline
from dagster.core.storage.fs_io_manager import fs_io_manager

import numpy as np


@solid
def test_solid(context) -> Nothing:
    context.log.info("Starting test")
    # ! error here, filename expected first
    np.savez(np.random.uniform(size=(100, 1000)), "test.npz")
    yield Output(None)


def pipeline_factory(*args, **kwargs):
    @pipeline(
        mode_defs=[
            ModeDefinition(
                executor_defs=[multiprocess_executor],
                resource_defs={"io_manager": fs_io_manager},
            )
        ],
    )
    def simple_pipeline():
        test_solid()

    return simple_pipeline


if __name__ == "__main__":
    os.environ["DAGSTER_HOME"] = "~/.dagster/"
    reconstructable_pipeline = build_reconstructable_pipeline(
        "dagster_test",
        "pipeline_factory",
        (),
        {},
    )
    execute_pipeline(
        reconstructable_pipeline,
        instance=DagsterInstance.get(),
        run_config={
            "execution": {
                "multiprocess": {
                    "config": {
                        "max_concurrent": 2,
                    }
                }
            },
        },
    )
