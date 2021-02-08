def bahamas_attrs(snap_info, slice_axis, slice_size):
    """Return the hdf5 attributes for BAHAMAS slices."""
    return {

        'slice_axis': slice_axis,
        'slice_size': slice_size,
        'box_size': snap_info.boxsize,
        'a': snap_info.a,
        'h': snap_info.h,
    }


def bahamas_layout_properties(snap_info, maxshape):
    """Return the hdf5 file layout for BAHAMAS slices."""
    joint = {
        'Coordinates': {
            'shape': (3, 0),
            'maxshape': (3, maxshape),
            'dtype': float,
            'attrs': {
                'CGSConversionFactor': snap_info.cm_per_mpc,
                'aexp-scale-exponent': 1.0,
                'h-scale-exponent': -1.0,
            }},
        'Mass': {
            'shape': (0,),
            'maxshape': (maxshape,),
            'dtype': float,
            'attrs': {
                'CGSConversionFactor': snap_info.mass_unit * 1e-10,
                'aexp-scale-exponent': 0.0,
                'h-scale-exponent': 0.0,
            }},
    }

    properties = {
        0 : {
            'Temperature': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'CGSConversionFactor': 1.0,
                    'aexp-scale-exponent': 0.0,
                    'h-scale-exponent': 0.0,
                }},
            'Density': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'CGSConversionFactor': snap_info.rho_unit,
                    'aexp-scale-exponent': -3.0,
                    'h-scale-exponent': 2.0,
                }},
            'SmoothedMetallicity': {
                'shape': (0,),
                'maxshape': (maxshape,),
                'dtype': float,
                'attrs': {
                    'CGSConversionFactor': 1.0,
                    'aexp-scale-exponent': 0.0,
                    'h-scale-exponent': 0.0,
                }},
            **joint,
        },
        1 : {**joint},
        4 : {**joint},
        5 : {**joint}
    }
    return properties


def bahamas_properties_to_dsets(parttypes, properties):
    """Return the dataset matching each property in properties."""
    pass


def get_slice_layout(slice_num, parttypes, properties, attrs):
    """Generate the standard layout for our slice files.

    Parameters
    ----------
    slice_num : int
        slice number
    parttypes : [0, 1, 4, 5]
        particle types to include
    properties : dict
        dictionary with parttypes as keys
        - parttype : dict
            - 'property' : dict
                - 'shape' : tuple
                - 'maxshape' : tuple
                - 'dtype' : dtype
                - 'attrs' : dict
                    - 'attr' : value
    attrs : dict
        general attributes to add to root of hdf5 file

    Returns
    -------
    layout : dict
        layout for hdf5 file
    """
    layout = {'attrs': {}, 'dsets': {}}
    for attr, val in attrs.items():
        layout['attrs'][attr] = val
        layout['attrs']['slice_num'] = slice_num

    for parttype in parttypes:
        for prop, val in properties[parttype].items():
            layout['dsets'][f'PartType{parttype}/{prop}'] = val

    return layout
