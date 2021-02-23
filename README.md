# Usage
We provide a batch processing mode in `batch_run.py`. A TOML file
serves as the configuration of the `run_pipeline` function. To run the
pipeline, the config file and the files specified within it need to
adhere to the following specifications. We go through each of the
headers in the TOML file sequentially.

## `sims`
For the `sims` header, we expect that there is a `base_dir` that holds
all of the different simulation runs, which are specified in
`sim_dirs`. For our use case, all simulations are part of the BAHAMAS
suite, and, hence, of the `Gadget` type. Different `simulation_type`'s
can be specified, but they will require a modification in the
`batch_run.slice_sim` function and in `sims.your_simulation_type` to
work. Ensure that you are able to pass all of the needed information
to `slicing.slice_particle_list`, which will slice each simulation.

## `slices`
The sliced simulations should all adhere to the structure laid out in
`sims.slice_layout`. This ensures that all following calculations on
the sliced simulations are consistent regardless of the
`simulation_type`. The slices are saved to
`{save_dir}/{sim_dir}/{slice_axis}_s_{slice_size}_{snapshot:03d}_n{i:d}.hdf5`,
where `slice_axis` corresponds to one of the `slice_axes` specified
(the different dimensions of the 3D coordinates specified by the
simulation). The `slice_size` key specifies the thickness of the
slices (in units of the box size of the simulation).

## `maps`
We can generate 2D projected maps from the slice files that we save.
To create a map, the coordinates to center on need to be specified.
These are expected in the `{coords_dir}/{sim_dir}/` directory. They
should be saved in `hdf5` file format with a main dataset
`coordinates` that contains the map centers. Any secondary information
about the map locations, such as halo masses etc., can be saved in the
same `hdf5` file.


