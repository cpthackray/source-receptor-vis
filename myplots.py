from myconfig import DATAPATH
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from cartopy import feature as cfeature
import xarray as xr
import io
from maps import make_map_base, add_colorbar, add_gridded, add_points
from srtools import InfluenceFunction

plt.style.use('seaborn-deep')

influence_function = InfluenceFunction(f'{DATAPATH}placeholder_data.nc')
lons, lats = influence_function.get_lonslats()

legal_congeners = ['PFHXA', 'PFHPA', 'PFPA', 'PFBUA', 'PFPRA', 'TFA']
legal_congener_names = ['PFHxA', 'PFHpA', 'PFPA', 'PFBA', 'PFPrA', 'TFA']
congname_lookup = {c: n for c, n in zip(legal_congeners, legal_congener_names)}
legal_emistypes = ['uniform', 'hazwaste', 'incinerator', 'landfill', 'wwtp',
                   'population']
legal_emisnames = ['Uniformly distributed', 'Hazardous waste disposal',
                   'Incinerators', 'Landfills', 'Wastewater treatment',
                   'Population-weighted']


def plot_dep(congener, E={'population': 1.0}, latlist=[], lonlist=[]):
    congs = [congener]
    xshift = -10

    cong = congener

    pdata = influence_function.get_species_exposure(E, congener, 'deposition')

    ax = make_map_base(projection='platecarree',
                       proj_args={'extent': (-128., -62.+xshift/2, 24., 48.)},
                       figsize=(12, 8), land='#b6a6a3')
    cg = add_gridded(ax, lons[:xshift], lats, np.log10(pdata)[:, :xshift], cmap='jet',
                     vmin=-2, vmax=np.log10(np.max(pdata)), gridlines='gray')
    if len(latlist) > 0:
        ax.scatter([float(x) for x in lonlist], [float(y) for y in latlist], marker='s',
                   s=20, color='k',
                   transform=ccrs.PlateCarree(), zorder=1000)

    add_colorbar(
        ax, cg, clabel='Deposition [ng m$^{-2}$ yr$^{-1}$]', logcolor=True, nticks=6)

    plt.title(f"{congname_lookup[cong]} Total deposition", fontsize=15)
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png', bbox_inches='tight')
    bytes_image.seek(0)
    return bytes_image


def plot_dep_point(latlist, lonlist, congener, E):
    if (len(latlist) < 1) or (len(lonlist) < 1):
        plt.figure(figsize=(7.5, 5))
    else:
        data = {}
        for emis in legal_emistypes:
            data[emis] = []
            name = emis + '_ole' + '_' + congener
            griddata = influence_function.get_species_exposure({emis: E.get(emis, 0)},
                                                               congener, 'deposition')
            for lat, lon in zip(latlist, lonlist):
                i = np.argmin(np.abs(int(lon)-lons))
                j = np.argmin(np.abs(int(lat)-lats))
                data[emis].append(griddata[j, i])
        plt.figure(figsize=(7.5, 5))
        x = range(len(latlist))
        bartops = np.zeros_like(x).tolist()
        for i, name in enumerate(legal_emistypes):
            newdata = data[name]
            plt.bar(x, newdata, bottom=bartops, label=legal_emisnames[i],
                    edgecolor='k')
            bartops = np.add(newdata, bartops).tolist()
        plt.xticks(x, [f'{lat}N, {lon}E' for lat, lon in zip(latlist, lonlist)],
                   rotation=45, fontsize=15)
        plt.ylabel("Local Deposition [ng m$^{-2}$ yr$^{-1}$]",
                   fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(0, np.max(bartops)*1.2)
        plt.xlim(-1, len(bartops)+1)
        plt.legend(title='Source type', bbox_to_anchor=(
            1.05, 1), loc='upper left')
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png', bbox_inches='tight')
    bytes_image.seek(0)
    return bytes_image


if __name__ == '__main__':
    congener = 'PFHXA'
    emisname = 'uniform'
    E = {'uniform': 1.0}
    plot_dep(congener, E)
    plt.show()
