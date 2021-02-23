from pathlib import Path

import numpy as np
import toml


class Config(object):
    def __init__(self, config_file):
        config = toml.load(config_file)
        self.base_dir = config['sims']['base_dir']
        self.sim_dirs = config['sims']['sim_dirs']
        self.sim_type = config['sims']['sim_type']
        self.snapshots = config['sims']['snapshots']

        self.slice_size = config['slices']['slice_size']
        self.slice_axes = config['slices']['slice_axes']
        self.slice_dir = config['slices']['save_dir']


        self.map_dir = config['maps']['save_dir']
        self.coords_name = config['maps']['coords_name']
        self.map_types = config['maps']['map_types']
        self.map_size = config['maps']['map_size']
        self.map_res = config['maps']['map_res']
        self.map_thickness = config['maps']['map_thickness']

        self.obs_dir = config['observables']['save_dir']

        self.build_config()

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
    def coords_name(self):
        return self._coords_name

    @coords_name.setter
    def coords_name(self, val):
        self._coords_name = val
        self.coords_files = [sd / f'{val}.hdf5' for sd in self.map_paths]

    @property
    def obs_dir(self):
        return self._obs_dir

    @obs_dir.setter
    def obs_dir(self, val):
        self._obs_dir = Path(val)
        self.obs_paths = [self._obs_dir / sd for sd in self.sim_dirs]

    @property
    def sim_type(self):
        return self._sim_type

    @sim_type.setter
    def sim_type(self, val):
        if val == 'BAHAMAS':
            self._sim_type = val
        else:
            raise ValueError(f'{val} is not a valid sim_type')

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, val):
        val = np.array(val, dtype=object)
        if len(val.shape) == 1:
            if val.shape[0] == 1:
                self._snapshots = np.ones(self._n_sims) * val
            elif val.shape[0] == self._n_sims:
                self._snapshots = val
            else:
                raise ValueError('snapshots should be scalar or list of length sim_dirs')
        elif len(val.shape) == 2:
            if val.shape[0] == self._n_sims:
                self._snapshots = val
            else:
                raise ValueError('snapshots should match len(sim_dirs) along dimension 0')
        else:
            raise ValueError('cannot match snapshots to sim_dirs')

    @property
    def slice_size(self):
        return self._slice_size

    @slice_size.setter
    def slice_size(self, val):
        val = np.array(val, dtype=object)
        if len(val.shape) == 1:
            if val.shape[0] == 1:
                self._slice_size = np.ones(self._n_sims) * val
            elif val.shape[0] == self._n_sims:
                self._slice_size = val
            else:
                raise ValueError('slice_size should be scalar or list of length sim_dirs')
        elif len(val.shape) == 2:
            if val.shape[0] == self._n_sims:
                self._slice_size = val
            else:
                raise ValueError('slice_size should match len(sim_dirs) along dimension 0')
        else:
            raise ValueError('cannot match slice_size to sim_dirs')

    @property
    def slice_axes(self):
        return self._slice_axes

    @slice_axes.setter
    def slice_axes(self, val):
        if set(val) & set([0, 1, 2]) != set(val):
            raise ValueError('slice_axes can only contain 0, 1, 2')
        else:
            self._slice_axes = np.atleast_1d(list(set(val)))

    @property
    def map_types(self):
        return self._map_types

    @map_types.setter
    def map_types(self, val):
        if set(val) & set(['gas_mass', 'dm_mass', 'stars_mass', 'bh_mass', 'sz']) != set(val):
            raise ValueError('map_types can only contain 0, 1, 2')
        else:
            self._map_types = np.atleast_1d(list(set(val)))

    def build_config(self):
        self.config = dict(
            dict(
                (
                    str(sim), dict(
                        (
                            ('snapshots', self.snapshots[idx]),
                            ('path', self.sim_paths[idx]),
                            ('slice_dir', self.slice_paths[idx]),
                            ('map_dir', self.map_paths[idx]),
                            ('coords_file', self.coords_files[idx]),
                            ('obs_dir', self.obs_paths[idx]),
                        )
                    )
                ) for idx, sim in enumerate(self.sim_dirs)
            )
        )
