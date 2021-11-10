import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from plotting_tools.axes import set_spines_labels

from map_sims import Config


FIGURE_DIR = Config().figure_dir
mpl.use('pdf')


def save_mass_map(
        mp, map_size, map_res, map_thickness, ptypes,
        coord, mass, group_id,
        coord_format=r'.2f', coord_unit=r'h^{-1} \, \mathrm{Mpc}',
        mass_label=r'\log_{10}m_\mathrm{200m}',
        mass_format=r'.2f',
        mass_unit=r'h^{-1} \, \mathrm{M_\odot}',
        map_label=r'\log_{10} \Sigma',
        map_unit=r'h \, \mathrm{M_\odot/Mpc^2}'):
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
    group_id : int
        id of the group
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
    dx = 0.8 / num_plots

    plt.clf()
    fig = plt.figure(figsize=(num_plots * 7, 8))

    axs = [
        fig.add_axes([0.1 + i * dx, 0.175, dx, 0.8])
        for i in range(num_plots)
    ]
    ax_cb = fig.add_axes([0.5 - dx, 0.075, 2 * dx, 0.1])

    cmap = mpl.cm.magma
    cmap.set_bad(color='black')

    # ensure valid colorbar range
    mp_tot = np.log10(mp.sum(axis=0))
    mp = np.log10(mp)

    vmin = np.quantile(mp_tot[~np.isneginf(mp_tot)], 0.01)
    vmax = np.quantile(mp_tot[~np.isneginf(mp_tot)], 0.99)

    img = axs[0].imshow(
        mp_tot, extent=(
            -map_size / 2, map_size / 2,
            -map_size / 2, map_size / 2),
        vmin=vmin, vmax=vmax,
        cmap=cmap, aspect='equal'
    )
    axs[0].set_title(r'total')
    axs[0].text(
        0.5, 0.05,
        f'${mass_label} = {format(mass, mass_format)} \, [{mass_unit}]$',
        ha='center', va='bottom',
        transform=axs[0].transAxes, color='white', fontsize=30,
    )
    set_spines_labels(
        axs[0], left=False, right=False, top=False, bottom=False, labels=False
    )

    scalebar = AnchoredSizeBar(
        axs[0].transData,
        map_size / 4, f'${map_size / 4} \, {coord_unit}$', 'upper left',
        pad=0.1, color='white',
        frameon=False, size_vertical=map_size / 100)
    axs[0].add_artist(scalebar)

    cb = plt.colorbar(img, cax=ax_cb, orientation='horizontal')
    cb.set_label(
        f'${map_label} \, [{map_unit}]$', color='white', labelpad=-85)

    for idx, ptype in enumerate(ptypes):
        i = idx + 1
        axs[i].imshow(
            mp[i - 1], extent=(
                -map_size / 2, map_size / 2,
                -map_size / 2, map_size / 2),
            vmin=vmin, vmax=vmax,
            cmap=cmap, aspect='equal'
        )
        axs[i].set_title(ptype_labels[ptype])
        set_spines_labels(
            axs[i], left=False, right=False, top=False,
            bottom=False, labels=False
        )

    axs[-1].text(
        0.5, 0.05,
        f'$\Delta l = {format(map_thickness, coord_format)} \, [{coord_unit}]$',
        ha='center', va='bottom',
        transform=axs[-1].transAxes, color='white', fontsize=30,
    )

    plt.savefig(f'{FIGURE_DIR}/map_gid_{group_id}.pdf')


def plot_masses(
        m_x, m_y,
        x_label='$\log_{10} m_\mathrm{200m} \, [h^{-1} \mathrm{M_\odot}]$',
        y_label='$\log_{10} M_\mathrm{\zeta}(R) \, [h^{-1} \mathrm{M_\odot}]$',
        text='$R = 0.5 \, h^{-1} \, \mathrm{Mpc}$'):
    """Plot masses"""
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    m_bins = np.linspace(m_x.min(), m_x.max(), 15)
    m_med = np.quantile([])
    ax.plot(m_x, m_y, lw=0, marker='o')
    ax.text(
        0.95, 0.05, text, ha='right', va='center',
        transform=ax.transAxes
    )
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.savefig(f'{FIGURE_DIR}/masses.pdf')
