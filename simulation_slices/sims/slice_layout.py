def attrs(num_slices, slice_axis, slice_size, box_size, snapshot):
    """Return the required hdf5 attributes for all slices."""
    return {
        'num_slices': num_slices,
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
                'description': 'Particle coordinates in cMpc / h',
                'single_value': False
            },
        },
        'masses': {
            'shape': (0,),
            'maxshape': (maxshape,),
            'dtype': float,
            'attrs': {
                'description': 'Particle masses M_sun / h',
                'single_value': False
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
                    'description': 'Particle temperatures in K',
                    'single_value': False
                }
            },
            'densities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle mass density in h^2 M_sun/Mpc^3',
                    'single_value': False
                }
            },
            'electron_number_densities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle electron number density',
                    'single_value': False
                }
            },
            'emissivities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle X-ray emissivity, L = 10**emissivity * sigma_sb * T^4',
                    'single_value': False
                }
            },
            **joint,
        },
        'dm': {
            **joint,
            'masses': {
                'shape': (0,),
                'maxshape': (1,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle masses M_sun / h',
                    'single_value': True
                },
            },
        },
        'stars': {**joint},
        'bh': {**joint},
    }
    return properties


def get_slice_layout(
        num_slices, slice_axis, slice_size, maxshape,
        box_size, snapshot, ptypes):
    """Generate the standard layout for our slice hdf5 files.

    Parameters
    ----------
    num_slices : int
        total number of slices
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
        num_slices=num_slices, slice_axis=slice_axis,
        slice_size=slice_size, box_size=box_size, snapshot=snapshot
    )

    props = properties(
        maxshape=maxshape
    )
    for ptype in valid_ptypes:
        for slice_idx in range(num_slices):
            for prop, val in props[ptype].items():
                layout['dsets'][f'{slice_idx}/{ptype}/{prop}'] = val

    return layout
