from pathlib import Path

import astropy.units as u
import numpy as np
import toml

import simulation_slices.maps.observables as obs


CONFIG_FILE = str(Path(__file__).parent / "batch.toml")


class Config(object):
    def __init__(self, config_file=CONFIG_FILE):
        self.config_file = config_file
        config = toml.load(config_file)
        self.logging = config["setup"]["logging"]
        if self.logging:
            self.log_dir = config["setup"]["log_dir"]
            self.log_level = config["setup"]["log_level"]
            self.log_name_append = config["setup"].get("log_name_append", "")

        # configuration for slice_sim
        self.base_dir = config["sims"]["base_dir"]
        self.sim_dirs = config["sims"]["sim_dirs"]
        self.sim_suite = config["sims"]["sim_suite"]
        self.snapshots = config["sims"]["snapshots"]
        self.ptypes = config["sims"]["ptypes"]
        self.box_sizes = config["sims"]["box_sizes"] * u.Unit(
            config["sims"]["box_sizes_units"]
        )

        self.num_slices = config["slices"]["num_slices"]
        self.slice_axes = config["slices"]["slice_axes"]
        self.slice_dir = config["slices"]["save_dir"]

        # configuration for save_coords
        mass_units = config["coords"].get("mass_units", None)
        log10_mass_range = config["coords"].get("log10_mass_range", None)
        if log10_mass_range is not None:
            self.mass_range = 10**np.array(log10_mass_range) * u.Unit(mass_units)

        self.coords_dir = config["coords"].get("coords_dir", None)
        self.coords_name = config["coords"].get("coords_name", None)
        self.coord_dset = config["coords"].get("coord_dset", None)
        self.mass_dset = config["coords"].get("mass_dset", None)
        self.extra_dsets = config["coords"].get("extra_dsets", None)
        if "sample_haloes_bins" in config["coords"].keys():
            sample_haloes_bins = config["coords"]["sample_haloes_bins"]
            n_bins = sample_haloes_bins["n_bins"]
            self.sample_haloes_bins = {
                "mass_bin_edges": 10**np.linspace(
                    *sample_haloes_bins["log10_mass_range"], n_bins + 1) * u.Unit(mass_units),
                "n_in_bin": np.ones(n_bins, dtype=int) * sample_haloes_bins["n_in_bin"]
            }
        else:
            self.sample_haloes_bins = None


        # configuration for map_sim
        self.map_dir = config["maps"]["save_dir"]
        self.map_name_append = config["maps"].get("map_name_append", "")
        self.map_method = config["maps"].get("map_method", None)
        self.map_types = config["maps"]["map_types"]
        self.map_pix = config["maps"]["map_pix"]
        self.map_size = config["maps"]["map_size"] * u.Unit(config["maps"]["map_units"])
        self.map_thickness = config["maps"]["map_thickness"] * u.Unit(
            config["maps"]["map_units"]
        )
        self.n_ngb = config["maps"].get("n_ngb", None)

        # configuration for observables
        self.obs_dir = config["observables"]["save_dir"]
        self.obs_types = config["observables"].keys() - ["save_dir"]
        self.obs_kwargs = {key: config["observables"][key] for key in self.obs_types}

        # self.figure_dir = config["DIRECTORIES"]["FIGURE_DIR"]
        # self.data_dir = config["DIRECTORIES"]["DATA_DIR"]

        self.build_config()

    def __getitem__(self, key):
        return self.config[key]

    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, val):
        path = Path(val)
        # if not path.exists():
        #     raise ValueError(f'{path} does not exist')
        self._base_dir = path

    @property
    def sim_dirs(self):
        return self._sim_dirs

    @sim_dirs.setter
    def sim_dirs(self, val):
        self._n_sims = len(val)
        self._sim_dirs = [Path(sd) for sd in val]
        self.sim_paths = [self.base_dir / sd for sd in val]

    @property
    def slice_dir(self):
        return self._slice_dir

    @slice_dir.setter
    def slice_dir(self, val):
        self._slice_dir = Path(val)
        self.slice_paths = [self._slice_dir / sd for sd in self.sim_dirs]

    @property
    def map_dir(self):
        return self._map_dir

    @map_dir.setter
    def map_dir(self, val):
        self._map_dir = Path(val)
        self.map_paths = [self._map_dir / sd for sd in self.sim_dirs]

    @property
    def coords_dir(self):
        return self._coords_dir

    @coords_dir.setter
    def coords_dir(self, val):
        self._coords_dir = Path(val)
        self.coords_paths = [self._coords_dir / sd for sd in self.sim_dirs]

    @property
    def coords_name(self):
        return self._coords_name

    @coords_name.setter
    def coords_name(self, val):
        self._coords_name = val
        self.coords_files = [
            [
                sd / f"{val}_{snap:03d}.hdf5" for snap in self.snapshots[sim_idx]
            ]  for sim_idx, sd in enumerate(self.coords_paths)
        ]

    @property
    def obs_dir(self):
        return self._obs_dir

    @obs_dir.setter
    def obs_dir(self, val):
        self._obs_dir = Path(val)
        self.obs_paths = [self._obs_dir / sd for sd in self.sim_dirs]

    @property
    def sim_suite(self):
        return self._sim_suite

    @sim_suite.setter
    def sim_suite(self, val):
        if val.lower() == "bahamas" or val.lower() == "miratitan":
            self._sim_suite = val
        else:
            raise ValueError(f"{val} is not a valid sim_suite")

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, val):
        if type(val) is list:
            # snapshots specified for each sim
            if len(val) == self._n_sims:
                self._snapshots = [np.atleast_1d(v) for v in val]
            # multiple snapshots for each sim
            else:
                self._snapshots = np.tile(np.atleast_1d(val)[None], (self._n_sims, 1))

        elif type(val) is int:
            self._snapshots = np.ones((self._n_sims, 1), dtype=int) * val
        else:
            raise ValueError("snapshots should be list or int")

    @property
    def ptypes(self):
        return self._ptypes

    @ptypes.setter
    def ptypes(self, val):
        if type(val) is list:
            # ptypes specified for each sim
            if len(val) == self._n_sims:
                self._ptypes = [np.atleast_1d(v) for v in val]
            # multiple ptypes for each sim
            else:
                self._ptypes = np.tile(np.atleast_1d(val)[None], (self._n_sims, 1))

        elif type(val) is str:
            self._ptypes = np.chararray(
                (self._n_sims, 1), itemsize=len(val), unicode=True
            )
            self._ptypes[:] = val

        else:
            raise ValueError("ptypes should be list or str")

    @property
    def box_sizes(self):
        return self._box_sizes

    @box_sizes.setter
    def box_sizes(self, val):
        if type(val) is list:
            # box_sizes specified for each sim
            if len(val) == self._n_sims:
                self._box_sizes = [np.atleast_1d(v) for v in val]
            else:
                raise ValueError("can only have 1 box_size per sim")

        elif type(val) is u.Quantity:
            self._box_sizes = np.ones(self._n_sims) * val
        else:
            raise ValueError("box_sizes should be list or astropy.units.Quantity")

    @property
    def num_slices(self):
        return self._num_slices

    @num_slices.setter
    def num_slices(self, val):
        if type(val) is not int:
            raise ValueError("num_slices should be int")
        self._num_slices = val

    @property
    def slice_axes(self):
        return self._slice_axes

    @slice_axes.setter
    def slice_axes(self, val):
        if set(val) & set([0, 1, 2]) != set(val):
            raise ValueError("slice_axes can only contain 0, 1, 2")
        else:
            self._slice_axes = np.atleast_1d(list(set(val)))

    @property
    def map_types(self):
        return self._map_types

    @map_types.setter
    def map_types(self, val):
        valid_map_types = obs.MAP_TYPES_OPTIONS.keys()
        if type(val) is list:
            # map_types specified for each sim
            if len(val) == self._n_sims:
                try:
                    if set(val) & set(valid_map_types) != set(val):
                        raise ValueError(
                            f"map_types can only contain {valid_map_types}"
                        )
                    else:
                        self._map_types = [[v] for v in val]
                # if val is list of lists
                except TypeError:
                    map_types = []
                    for v in val:
                        if set(v) & set(valid_map_types) != set(v):
                            raise ValueError(
                                f"map_types can only contain {valid_map_types}"
                            )
                        else:
                            map_types.append(v)
                    self._map_types = map_types

            # multiple map_types for each sim
            else:
                self._map_types = np.tile(
                    np.atleast_1d(val)[None], (self._n_sims, 1)
                ).tolist()

        elif type(val) is str:
            self._map_types = np.asarray([val] * self._n_sims).tolist()
        else:
            raise ValueError("map_types should be list or string")

        # if set(val) & set(['gas_mass', 'dm_mass', 'stars_mass', 'bh_mass', 'sz']) != set(val):
        #     raise ValueError('map_types can only contain 0, 1, 2')
        # else:
        #     self._map_types = np.atleast_1d(list(set(val)))

    def build_config(self):
        self.config = dict(
            dict(
                (
                    str(sim),
                    dict(
                        (
                            ("snapshots", self.snapshots[idx]),
                            ("path", self.sim_paths[idx]),
                            ("slice_dir", self.slice_paths[idx]),
                            ("map_dir", self.map_paths[idx]),
                            ("coords_file", self.coords_files[idx]),
                            ("obs_dir", self.obs_paths[idx]),
                        )
                    ),
                )
                for idx, sim in enumerate(self.sim_dirs)
            )
        )
