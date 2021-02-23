def attrs(slice_num, slice_axis, slice_size, box_size, snapshot):
    """Return the required hdf5 attributes for all slices."""
    return {
        'slice_num': slice_num,
        'slice_axis': slice_axis,
        'slice_size': slice_size,
        'box_size': box_size,
        'snapshot': snapshot
    }


def properties(maxshape):
    """Return the standard properties expected for the slice file."""
    joint = {
        'coordinates': {
            'shape': (3, 0),
            'maxshape': (3, maxshape),
            'dtype': float,
            'attrs': {
                'description': 'Particle coordinates in Mpc / h'
            },
        },
        'masses': {
            'shape': (0,),
            'maxshape': (maxshape,),
            'dtype': float,
            'attrs': {
                'description': 'Particle masses M_sun / h'
            },
        }
    }
    properties = {
        'gas': {
            'temperatures': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle temperatures in K'
                }
            },
            'densities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle mass density in h^2 M_sun/Mpc^3'
                }
            },
            'electron_number_densities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle electron number density'
                }
            },
            # # required to compute electron density for SZ signal
            # 'smoothed_hydrogen': {
            #     'shape': (0,),
            #     'maxshape': (maxshape,),
            #     'dtype': float,
            #     'attrs': {
            #         'description': 'Particle Hydrogen mass fraction'
            #     }
            # },
            # 'smoothed_helium': {
            #     'shape': (0,),
            #     'maxshape': (maxshape,),
            #     'dtype': float,
            #     'attrs': {
            #         'description': 'Particle Helium mass fraction'
            #     }
            # },
            **joint,
        },
        'dm': {**joint},
        'stars': {**joint},
        'bh': {**joint},
    }
    return properties


def get_slice_layout(
        slice_num, slice_axis, slice_size, maxshape,
        box_size, snapshot, ptypes):
    """Generate the standard layout for our slice hdf5 files.

    Parameters
    ----------n
    slice_num : int
        slice number
    slice_axis : int
        dimension along which box has been sliced
    slice_size : float
        size of the slices
    maxshape : int
        maximum shape for each hdf5 dataset
    box_size : float
        box size
    snapshot : int
        snapshot of the simulation
    ptypes : ['gas', 'dm', 'stars', 'bh']
        particle types to include

    Returns
    -------
    layout : dict
        layout for hdf5 file
    """
    ptype_options = ['gas', 'dm', 'stars', 'bh']
    valid_ptypes = set([p.lower() for p in ptypes]) & set(ptype_options)

    layout = {'dsets': {}}
    layout['attrs'] = attrs(
        slice_num=slice_num, slice_axis=slice_axis,
        slice_size=slice_size, box_size=box_size, snapshot=snapshot
    )

    props = properties(
        maxshape=maxshape
    )
    for ptype in valid_ptypes:
        for prop, val in props[ptype].items():
            layout['dsets'][f'{ptype}/{prop}'] = val

    return layout
