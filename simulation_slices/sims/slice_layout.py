def attrs(
        num_slices: int,
        slice_axis: int,
        slice_size: int,
        box_size,
        snapshot, z, a):
    """Return the required hdf5 attributes for all slices."""
    return {
        'num_slices': num_slices,
        'slice_axis': slice_axis,
        'slice_size': slice_size,
        'box_size': box_size,
        'snapshot': snapshot,
        'a': a,
        'z': z,
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
                'single_value': False,
                'units': 'Mpc / littleh'
            },
        },
        'masses': {
            'shape': (0,),
            'maxshape': (maxshape,),
            'dtype': float,
            'attrs': {
                'description': 'Particle masses M_sun / h',
                'single_value': False,
                'units': 'solMass / littleh',
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
                    'single_value': False,
                    'units': 'K',
                }
            },
            'densities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle mass density in h^2 M_sun/Mpc^3',
                    'single_value': False,
                    'units': 'littleh2 solMass / Mpc3',
                }
            },
            'electron_number_densities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle electron number density',
                    'single_value': False,
                    'units': 'littleh2 / Mpc3',
                }
            },
            'luminosities': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'description': 'Particle X-ray luminosity in L_sun',
                    'single_value': False,
                    'units': 'solLum',
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
                    'single_value': True,
                    'units': 'solMass / littleh',
                },
            },
        },
        'stars': {**joint},
        'bh': {**joint},
    }
    return properties


def get_slice_layout(
        num_slices, slice_axis, slice_size, maxshape,
        box_size, snapshot, z, a, ptypes):
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
    z : float
        redshift
    a : float
        expansion factor
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
        num_slices=num_slices, slice_axis=slice_axis, z=z, a=a,
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
