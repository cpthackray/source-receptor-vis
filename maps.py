import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from copy import copy
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from cartopy import feature as cfeature
from cartopy.io import shapereader


def make_map(projection="Robinson",
             proj_args={},
             logcolor=False,
             scatter_color=True,
             scatter_size=None,
             clabel='',
             ax=None,
             gridlines='gray',
             coastlines='k',
             land='#964B00',
             ocean='#006994',
             statelines=None,
             natborders='gray',
             lakelines=None,
             lakes='#006994',
             **kwargs):
    if pointdata is not None:
        plons, plats, pdata = pointdata[0, :], pointdata[1, :], pointdata[2, :]
        pmax, pmin = np.nanmax(pdata), np.nanmin(pdata)
    if gridded is not None:
        gmax, gmin = np.nanmax(gridded), np.nanmin(gridded)

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

    ourcmap = kwargs.get('cmap', 'jet')
    thecmap = copy(matplotlib.cm.get_cmap(ourcmap))
    thecmap.set_bad(color='w', alpha=0.)


def make_map_base(projection="Robinson",
                  proj_args={},
                  coastlines='k',
                  land='#964B00',
                  ocean='#006994',
                  statelines=None,
                  natborders='gray',
                  lakelines=None,
                  lakes='#006994',
                  **kwargs):
    """Plot gridded data on map.
    Optionally add pointdata sequence of (lon,lat,data)."""
    cl = proj_args.get('central_longitude', 180)
    extent = proj_args.get('extent', None)
    if projection.lower() in ['mollweide']:
        proj = ccrs.Mollweide(central_longitude=cl)
    elif projection.lower() in ['platecarree', 'flat']:
        proj = ccrs.PlateCarree(central_longitude=cl)
    elif projection.lower() in ['eckertiv']:
        proj = ccrs.EckertIV(central_longitude=cl)
    elif projection.lower() in ['robinson']:
        proj = ccrs.Robinson(central_longitude=cl)
    else:
        print("IMPLEMENT PROJECTION!")

    figsize = kwargs.get('figsize', (12, 8))
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)  # ccrs.PlateCarree())

    if coastlines is not None:
        ax.coastlines(color=coastlines)
    # ax.gridlines()
    if statelines is not None:
        ax.add_feature(
            cfeature.NaturalEarthFeature('cultural',
                                         'admin_1_states_provinces_lines',
                                         '50m',
                                         edgecolor=statelines,
                                         facecolor='none',
                                         linestyle='-'))
    if natborders is not None:
        ax.add_feature(cfeature.BORDERS, color=natborders)
    if (lakelines is not None) or (lakes is not None):
        ax.add_feature(cfeature.LAKES, edgecolor=lakelines, facecolor=lakes)
    if land is not None:
        # np.array( [0.9375, 0.9375, 0.859375]))
        ax.add_feature(cfeature.LAND, color=land)
    if ocean is not None:
        ax.add_feature(cfeature.OCEAN, color=ocean)

    ftitle = kwargs.get('title', '')
    ax.set_title(ftitle, fontsize=15)

    if extent is not None:
        ax.set_extent(extent)
        ratio = abs((extent[0] - extent[1]) / (extent[2] - extent[3]))
    else:
        ratio = 1.0
    return ax


def add_gridded(ax, glons, glats, gridded, gridlines=None, **kwargs):
    vmax = kwargs.get('vmax', np.max(gridded))
    vmin = kwargs.get('vmin', np.min(gridded))
    thecmap = kwargs.get('cmap', 'jet')

    c = ax.pcolormesh(glons,
                      glats,
                      gridded,
                      vmax=vmax,
                      vmin=vmin,
                      linewidth=0.5,
                      edgecolor=gridlines,
                      antialiased=True,
                      transform=ccrs.PlateCarree(),
                      cmap=thecmap)
    return c


def add_points(ax, plons, plats, points, scatter_size=50, **kwargs):
    vmax = kwargs.get('vmax', np.max(points))
    vmin = kwargs.get('vmin', np.min(points))
    thecmap = kwargs.get('cmap', 'jet')
    edgecolor = kwargs.get('edgecolor', 'k')

    if scatter_size is not None:
        ss = scatter_size
    else:
        ss = 40
    c = plt.scatter(plons,
                    plats,
                    marker='o',
                    norm=Normalize(vmin, vmax),
                    transform=ccrs.PlateCarree(),
                    c=points,
                    s=ss,
                    zorder=1000,
                    edgecolor=edgecolor,
                    cmap=thecmap)
    return c


def add_colorbar(ax, c, logcolor=False, orientation='horizontal',
                 clabel='', ratio=1.0, nticks=7, **kwargs):

    vmax = kwargs.get('vmax', c.norm.vmax)
    vmin = kwargs.get('vmin', c.norm.vmin)

    if logcolor:
        cticks = np.linspace(vmin, vmax, nticks)
        cticklabels = []
        for x in cticks:
            if x < -3:
                cticklabels.append(f'{10**x:.1e}')
            elif x < -2:
                cticklabels.append(f'{10**x:.3f}')
            elif x < 0:
                cticklabels.append(f'{10**x:.2f}')
            elif x < 4:
                cticklabels.append(f'{10**x:.0f}')
            else:
                cticklabels.append(f'{10**x:.1e}')
        #cticklabels = [f'{10**x:.2f}' for x in cticks]
    else:
        cticks = np.linspace(vmin, vmax, nticks)
        cticklabels = []
        for x in cticks:

            m = np.log10(abs(x))
            #s = np.sign(x)
            if m < -1e4:
                cticklabels.append('0')
            elif m < -3:
                cticklabels.append(f'{x:.1e}')
            elif m < -2:
                cticklabels.append(f'{x:.3f}')
            elif m < 0:
                cticklabels.append(f'{x:.2f}')
            elif m < 4:
                cticklabels.append(f'{x:.0f}')
            else:
                cticklabels.append(f'{x:.1e}')
        #cticklabels = [f'{x:.2f}' for x in cticks]
    if orientation == 'horizontal':
        frc = 0.056
    elif orientation == 'vertical':
        frc = 0.02
    cbar = plt.colorbar(c,
                        orientation=orientation,
                        fraction=frc * ratio,
                        pad=0.01,
                        ticks=cticks)
    cbar.set_label(clabel, fontsize=15)
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels(cticklabels, fontsize=15)
    elif orientation == 'vertical':
        cbar.ax.set_yticklabels(cticklabels, fontsize=15)


def add_features(ax, fname):
    shape_feature = cfeature.ShapelyFeature(shapereader.Reader(fname).geometries(),
                                            ccrs.PlateCarree(), facecolor='k')
    ax.add_feature(shape_feature)
