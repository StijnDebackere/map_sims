# `map_sims`

Tools for generating projected maps from cosmological hydrodynamical
simulations.

## Installation
Clone and install this repository by running
```
git clone https://github.com/StijnDebackere/map_sims
cd map_sims
pip install -e .
```

This will install command-line scripts that can generate the simulation maps

## Usage

Batch processing scripts to generate maps and to save halo metadata
are available directly from the command-line by running
```
map_sims config.toml --snapshots xxx yyy --project-full --save-info
```
or
```
slurm_map_sims --config config.toml --n-tasks n --log-dir /path/to/log/ --save-info --map-full
```
The `config.toml` file specifies the settings for the map generation (see `templates/batch_example.toml`).

Aperture masses can be calculated from the saved maps by running
```
extract_masses /path/to/map_names_file /path/to/info_names_file --sim_suite xxx --base_dir /path/to/base --radii-file radii.toml --overwrite
```
or
```
slurm_extract_masses /path/to/map_names_file /path/to/info_names_file --sim_suite xxx --base_dir /path/to/base --radii-file radii.toml --overwrite
```
with `map_names_file` and `info_names_file` being text files
containing the filenames for the simulation maps and metadata files
generated by `map_sims`. The `radii.toml` file specifies the aperture radii to use (see `templates/radii.toml`).

## Configuration
A TOML file serves as the configuration for the batch runs (for an
example, see the `templates/` directory). To run the pipeline, the
config file and the files specified within it need to adhere to the
following specifications. We go through each of the headers in the
TOML file sequentially.

### `sims`
For the `sims` header, we expect that there is a `base_dir` that holds
all of the different simulation runs, which are specified in
`sim_dirs`. The `sim_suite` should be defined in
`map_sims.sims.read_sim`, `MiraTitan` and `BAHAMAS` are available by
default.

Maps can be generated for a slew of different `snapshots` and
`slice_axes`. The simulation `box_size` and the `box_sizes_units` are
required, as are the different `ptypes` to extract.

### `info`
Additionally, simulation metadata can be saved to enable the
calculation of observables centred on identified halos. These files
are saved to `info_dir` under `info_name`. These files contain
minimally the halo group IDs, the halo coordinates, and the halo
masses. A `log10_mass_range` in `mass_units` can be specified to only
extract the observed signal for a subset of the simulated haloes.

### `maps`
The projected maps are saved to `save_dir`, with an option to append
extra information to the filename using `map_name_append`. We try to
be smart in saving data, appending additional information to the maps
if it has not been saved yet. However, files can be overwritten by
setting `map_overwrite` to true.

Maps can be generated by binning particles per pixel, using
`map_method="bin"`. SPH interpolation is also available (although
expensive) by setting `map_method="sph"`. The `map_types` are defined
in `map_sims.maps.observables` under the `MAP_TYPES_OPTIONS` global
variable. This dictionary lists the function `func` that computes the
different types of maps, specified as the keys, them along with the
required `ptype` and `properties` that need to be saved in `info`.

The saved maps have `map_pix`x`map_pix` pixels. If `map_full=false`, a
list of `map_thickness` can be specified to project smaller volumes
with respect to the midpoint of the `slice_axis`.
