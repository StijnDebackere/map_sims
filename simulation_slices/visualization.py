import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np


def save_mass_map(
        mp, map_size, map_res, map_thickness, ptypes,
        coord, coord_format=r'.2f', coord_unit=r'h^{-1} \, \mathrm{Mpc}',
        mass, mass_label=r'\log_{10}m_\mathrm{200m}',
        mass_format=r'.2f',
        mass_unit=r'h^{-1} \, \mathrm{M_odot}'):
    """Save an image of mass map mp.

    Parameters
    ----------
    mp : (num_ptypes, num_pix, num_pix) array
        mass map for each particle type
    map_size : float
        size of the map in units of box_size
    map_res : float
        resolution of the map in units of box_size
    map_thickness : float
        thickness of the map projection in units of box_size
    ptypes : (num_ptypes,) array
        particle types in mp
        [0=gas, 1=dm, 4=stars, 5=bh]
    coord : (3,) array
        central coordinate of the map in units of box_size
    mass : float
        chosen mass
    mass_label : str
        TeX-formatted label to give to mass
    mass_format : str
        format string for mass
    mass_unit : str
        TeX-formatted mass unit

    Returns
    -------
    saves an image of the mass map

    """
    ptype_labels = {
        0 : 'gas',
        1 : 'dm',
        4 : 'stars',
        5 : 'bh'
    }
    num_plots = len(ptypes) + 1
    plt.clf()
    fig, axs = plt.subplots(
        nrows=1, ncols=num_plots , sharey=True,
        gridspec_kw={
            'height_ratio': [1, 1],
        },
        figsize=(num_plots * 5, 5)
    )

    vmin = mp[0].min()
    vmax = mp[0].max()
    axs[0].imshow(
        mp[0], extent=(
            -map_size / 2, map_size / 2,
            -map_size / 2, map_size / 2),
        vmin=vmin, vmax=vmax,
        cmap='magma'
    )
    axs[0].set_title(r'total')
    axs[0].text(
        0.05, 0.05,
        f'${mass_label} = {format(mass, mass_format)} \, [{mass_unit}]$',
        transform=axs[0].transAxes, color='white', fontsize=30,
    )

    for idx, ptype in enumerate(ptypes):
        i = idx + 1
        axs[i].imshow(
            mp[i], extent=(
                -map_size / 2, map_size / 2,
                -map_size / 2, map_size / 2),
            vmin=vmin, vmax=vmax,
            cmap='magma'
        )

    axs[-1].text(
        0.05, 0.05,
        f'$\Delta l = {format(map_thickness, coord_format)} \, [{coord_unit}]$',
        transform=axs[-1].transAxes, color='white', fontsize=30,
    )
