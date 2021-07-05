from myconfig import DATAPATH
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from cartopy import feature as cfeature
import xarray as xr
import io

plt.style.use('seaborn-deep')


def plotmap(lons,
            lats,
            gridded=None,
            pointdata=None,
            projection="Mollweide",
            proj_args={},
            logcolor=False,
            clabel='',
            **kwargs):
    """Plot gridded data on map.
    Optionally add pointdata sequence of (lon,lat,data)."""
    cl = proj_args.get('central_longitude', 180)
    extent = proj_args.get('extent', None)
    if projection.lower() in ['mollweide']:
        proj = ccrs.Mollweide(central_longitude=cl)
    elif projection.lower() in ['platecarree', 'flat']:
        proj = ccrs.PlateCarree(central_longitude=cl)

    if pointdata is not None:
        plons, plats, pdata = pointdata[0, :], pointdata[1, :], pointdata[2, :]
        pmax, pmin = np.max(pdata), np.min(pdata)
    if gridded is not None:
        gmax, gmin = np.max(gridded), np.min(gridded)

    if (gridded is not None) and (pointdata is not None):
        vvmax = max(gmax, pmax)
        vvmin = min(gmin, pmin)
    elif gridded is not None:
        vvmax, vvmin = gmax, gmin
    elif pointdata is not None:
        vvmax, vvmin = pmax, pmin
    else:
        print('???')

    vvmin = kwargs.get('vmin', vvmin)
    vvmax = kwargs.get('vmax', vvmax)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=proj)  # ccrs.PlateCarree())
    ourcmap = kwargs.get('cmap', 'jet')
    if gridded is not None:
        c = ax.pcolormesh(lons,
                          lats,
                          gridded,
                          vmax=vvmax,
                          vmin=vvmin,
                          linewidth=0.5,
                          edgecolor='k',
                          antialiased=True,
                          transform=ccrs.PlateCarree(),
                          cmap=ourcmap)
    else:
        ax.pcolormesh(lons,
                      lats,
                      np.zeros((len(lats), len(lons))) + np.nan,
                      transform=ccrs.PlateCarree())
    if pointdata is not None:
        c = plt.scatter(plons,
                        plats,
                        marker='o',
                        norm=Normalize(vvmin, vvmax),
                        transform=ccrs.PlateCarree(),
                        c=pdata,
                        edgecolor='w',
                        cmap=ourcmap)

    ax.coastlines(color='w')
    # ax.gridlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural',
                                     'admin_1_states_provinces_lines',
                                     '50m',
                                     edgecolor='w',
                                     facecolor='none',
                                     linestyle='-'))
    ax.add_feature(cfeature.BORDERS, color='w')
    ax.add_feature(cfeature.LAKES, edgecolor='w', facecolor='none')

    ftitle = kwargs.get('title', '')
    ax.set_title(ftitle, fontsize=15)

    if extent is not None:
        ax.set_extent(extent)
        ratio = abs((extent[0] - extent[1]) / (extent[2] - extent[3]))
    else:
        ratio = 1.0

    if logcolor:
        cticks = np.arange(vvmin, vvmax, 0.5)
        cticklabels = [f'{10**x:.2f}' for x in cticks]
    else:
        cticks = np.linspace(vvmin, vvmax, 9)
        cticklabels = [f'{x:.2f}' for x in cticks]
    cbar = fig.colorbar(c,
                        orientation='horizontal',
                        fraction=0.046 * ratio,
                        pad=0.01,
                        ticks=cticks)
    cbar.set_label(clabel, fontsize=15)
    cbar.ax.set_xticklabels(cticklabels, fontsize=15)

    return fig


unitconv = 1e12 * 3600 * 24 * 365  # kg/s to ng/yr
MW = 500  # g/mol
unitconv_c = 1e12 * 2.5e25 * MW / 6.02e23  # mol/mol to pg/m3


def get_data(name, congs=["PFHXA"]):
    root = name.rsplit("_")[0]
    budget_ds = xr.open_dataset(
        f"{DATAPATH}archive/{name}/budget.nc")
    conc_ds = xr.open_dataset(
        f"{DATAPATH}archive/{name}/conc.nc")
    emis_ds = xr.open_dataset(
        f"{DATAPATH}archive/{root}/emis.nc")
    area = emis_ds['AREA'].values
    lats, lons = budget_ds['lat'].values, budget_ds['lon'].values

    emis_ngm2yr = emis_ds['PFAS_FT6'][0, 0, :,
                                      :].values * unitconv  # kg/s/cell to ng/m2/yr
    em_ss = (emis_ds['PFAS_PFOASSL'][0, 0, :, :].values +
             emis_ds['PFAS_PFOASSS'][0, 0, :, :].values) * unitconv
    conc_pgm3 = (conc_ds['SpeciesConc_FTOH62'][0, 0, :, :].values +
                 conc_ds['SpeciesConc_FTOLE62'][0, 0, :, :].values)*unitconv_c
    tote = np.sum(emis_ngm2yr*area) * 1e-12 * 1e-3  # ng/m2/yr to t/yr
    tote_ss = np.sum(em_ss*area) * 1e-12 * 1e-3  # ng/m2/yr to t/yr

    dep_ngm2yr, totd = {}, {}
    for cong in congs:
        dep_kgs = budget_ds[f'BudgetWetDepFull_{cong}'].values + budget_ds[f'BudgetWetDepFull_{cong}P'].values \
            + budget_ds[f'BudgetEmisDryDepFull_{cong}'].values + \
            budget_ds[f'BudgetEmisDryDepFull_{cong}P'].values
        totd[cong] = -np.sum(dep_kgs) * 3600 * 24 * 365 * 1e-3  # kg/s to t/yr
        dep_ngm2yr[cong] = unitconv * -np.mean(dep_kgs, axis=0) / area
        totd['sum'] = totd.get('sum', 0) + totd[cong]

    dep_kgs = budget_ds[f'BudgetWetDepFull_PFOASSL'].values + budget_ds[f'BudgetWetDepFull_PFOASSS'].values \
        + 0  # budget_ds[f'BudgetEmisDryDepFull_PFOASSL'].values + budget_ds[f'BudgetEmisDryDepFull_PFOASSS'].values \
    #- (emis_ds['PFAS_PFOASSL'][:,0,:,:].values + emis_ds['PFAS_PFOASSS'][:,0,:,:].values)
    dep_ss = unitconv * -np.mean(dep_kgs, axis=0) / area

    return dep_ngm2yr, emis_ngm2yr, totd, tote, conc_pgm3, dep_ss, tote_ss


conc_ds = xr.open_dataset('backend/placeholder_data.nc')
lats, lons = conc_ds['lat'].values, conc_ds['lon'].values

legal_congeners = ['PFHXA', 'PFHPA', 'PFPA', 'PFBUA', 'PFPRA', 'TFA']
legal_congener_names = ['PFHxA', 'PFHpA', 'PFPA', 'PFBA', 'PFPrA', 'TFA']
congname_lookup = {c: n for c, n in zip(legal_congeners, legal_congener_names)}
legal_emistypes = ['uniform', 'hazwaste', 'incinerator', 'landfill', 'wwtp',
                   'population']
legal_emisnames = ['Uniformly distributed', 'Hazardous waste disposal',
                   'Incinerators', 'Landfills', 'Wastewater treatment',
                   'Population-weighted']


def get_IF(congener, emisname):
    name = f'{emisname}_ole'
    ds = xr.open_dataset('backend/placeholder_data.nc')
    pdata = ds[name+'_'+congener].values[0, :, :]
    return pdata


def plot_dep(congener, E={'population': 1.0}):
    congs = [congener]
    xshift = -10

    cong = congener
    pdata = get_IF(congener, 'population') * 0.0
    for emisname, e in E.items():
        pdata += get_IF(congener, emisname) * e

    figd = plotmap(lons[:xshift], lats, np.log10(pdata)[:, :xshift], vmax=np.log10(np.max(pdata)),
                   vmin=-2, projection='platecarree',
                   clabel='Deposition [ng m$^{-2}$ yr$^{-1}$]',
                   title=f"{congname_lookup[cong]} Total deposition", logcolor=True,
                   proj_args={'extent': (-128., -62.+xshift/2, 24., 48.)})
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png', bbox_inches='tight')
    bytes_image.seek(0)
    return bytes_image


def plot_dep_point(latlist, lonlist, congener, E):
    if (len(latlist) < 1) or (len(lonlist) < 1):
        plt.figure(figsize=(7.5, 5))
    else:
        ds = xr.open_dataset('backend/placeholder_data.nc')
        lats, lons = ds['lat'].values, ds['lon'].values
        data = {}
        for emis in legal_emistypes:
            data[emis] = []
            name = emis + '_ole' + '_' + congener
            for lat, lon in zip(latlist, lonlist):
                i = np.argmin(np.abs(int(lon)-lons))
                j = np.argmin(np.abs(int(lat)-lats))
                data[emis].append(ds[name].values[0, j, i]*E.get(emis, 0))
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
