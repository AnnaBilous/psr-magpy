# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.spatial.transform import Rotation as Rot
from astropy.io import fits as pf
import colorsys, random
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

# import seaborn as sns
import healpy as hp
import sys
import os

# module_path = os.path.abspath(os.path.join('/media/anya/My\ Book/Multipole_OFL/'))
module_path = os.path.abspath(os.path.join("../../"))
sys.path.append(module_path)


from magn_lib import *
import colorsys as cl
import cartopy.crs as ccrs

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap

# from healpy.newvisufunc import  projview, newprojplot
# %%

Rns = 1
# Rmin = 5
# Rmax = 200

# grcl = '#596673'
# grcl = '#330000'
cap_coeff = (3**1.5) / 2.0
cap_coeff_plot = 1.25
mx = 6
Rns = 1  # Radius of neutron star
RLC = 7  # Radius of the light cylinder
K = 10  # Number of zorder layers per unit depth
XZview = RLC  # XZ viewport size
Nlayers = int(RLC * 2 * K * 2)  # Total number of zorder layers
zorder_layer = np.arange(0, Nlayers, 1)
# XZview = RLC*0.8 # XZ viewport size

end_layer = (
    Nlayers - 1
)  # Effectively, camera position. Layers with zorders above this (closest to the viewer) will not be rendered.
# end_layer = int(RLC*0.75*K) # Effectively, camera position. Layers with zorders above this (closest to the viewer) will not be rendered.

Nframes = 64  # Number of frames
rotph0 = 80  # Starting spin phase
# ph0 = 10 # Starting spin phase
# ph0 = 0 # Starting spin phase

lw_min = 0.07
lw_max = 0.5
lw_med = 0.2

pli = -np.log(lw_med - lw_min) / (lw_max - lw_min) / np.log(2)


def lw_func(layer_no):
    return (lw_max - lw_min) * (layer_no * 1.0 / Nlayers) ** pli + lw_min


def initialize_field_line_collection(NPgr, lws, cls):
    norm = plt.Normalize(
        -XZview, XZview
    )  # Normalizing zorder colors to the size of XZ viewport. Can be tweaked for aesthetics.
    lines = []
    # Initializing line collections for all zorder layers
    for j in range(NPgr):
        cur_line = []
        for k in np.arange(0, end_layer, 1):
            # line_layer = LineCollection(np.empty((0,2,2)), color=cls[j], zorder=zorder_layer[k],   linewidths=lws[j], antialiaseds=1, alpha=1.0, capstyle='round')
            line_layer = LineCollection(
                np.empty((0, 2, 2)),
                color=cls[j],
                zorder=zorder_layer[k],
                linewidths=lw_func(k),
                antialiaseds=1,
                alpha=1.0,
                capstyle="round",
            )
            cur_line.append(line_layer)
        lines.append(cur_line)
    return lines


def get_field_lines(savename, nline, Pgrid):

    NPgr = len(Pgrid)
    rlc_grid = 4.77 * Pgrid

    TH_all = np.copy(gr.th)
    PH_all = np.copy(gr.ph)
    V = np.zeros(nline)
    edges = []
    FLs_all = []

    Rvb = np.load(savename + "_FLstat.npy")
    ind = Rvb[:, 0].astype(int)
    cylmax = Rvb[:, 1]
    radmax = Rvb[:, 2]
    zmax = Rvb[:, 3]

    for rr in range(NPgr):
        cr = (cylmax > rlc_grid[rr]) | (
            (cylmax < rlc_grid[rr]) & (zmax > (cap_coeff * rlc_grid[rr]))
        )
        V[ind[cr]] = rr + 1

        with open(
            "/media/anya/8248818948817D23/Multipole_radio/FL_illustr_poster/FL_data/P_%02d_%02d_n%d_%02d.pickle"
            % (i, j, nside, rr),
            "rb",
        ) as tp:
            FLs_em = pickle.load(tp)

        edge = [FLe.num for FLe in FLs_em]
        edges.append(edge)

        FLs = []
        for FLe in FLs_em:
            FLs.append(FLe.FL[1:, :])

        FLs_all.append(FLs)
    V[V == 0] = np.NaN

    return FLs_all, V, edges


def fill_lines(FLs_all, NPgr, ph, incl, lines):
    # Rotating around z axis
    r1 = R.from_rotvec([np.deg2rad(ph) * np.array([0, 0, 1])])
    r2 = R.from_rotvec([np.deg2rad(incl) * np.array([1, 0, 0])])
    r = r2 * r1
    segments_all = []
    zorder_all = []
    for j in range(NPgr):
        FLs = FLs_all[j]
        # print(len(FLs))
        segments = np.empty((0, 2, 2))
        y = np.array([])
        Nline_cur = len(FLs)
        for k in range(Nline_cur):
            r_fl = FLs[k]
            X = r_fl[0, :]
            Y = r_fl[1, :]
            Z = r_fl[2, :]
            # Rch = (X**2+Y**2+Z**2)**0.5
            # print(np.max(Rch))
            # plt.plot(Y, Z, 'k')
            Xproj, Yproj, Zproj = (r.apply(np.transpose((X, Y, Z)))).T
            points = np.array([Xproj, Zproj]).T.reshape(-1, 1, 2)
            segments_cur = np.concatenate([points[:-1], points[1:]], axis=1)
            segments = np.append(segments, segments_cur, axis=0)
            y = np.append(y, 0.5 * (Yproj[1:] + Yproj[:-1]))
            zorder = ((y + 2 * RLC) * K).astype(int)  # zorder from 0 to Nlayers
        # plt.show()
        segments_all.append(segments)
        zorder_all.append(zorder)

    # Updating line collection data for each zorder layer
    for k in np.arange(0, end_layer, 1):
        for j in range(NPgr):
            cr = zorder_all[j] == zorder_layer[k]
            lines[j][k].set_segments(segments_all[j][cr, :, :])

    return lines


nside = 128
gr = grid(kind="healpix", nside=nside)
dirn = "../../Run_03_paper_illustr/"
plotdir = "/media/anya/8248818948817D23/Multipole_radio/FL_illustr_poster/"
Pgrid = np.array([2 / 4.77, 1.3, 4.9])  # 7.])
# lws = np.array([0.03, 0.06, 0.06, 0.1])
lws = np.array([0.03, 0.06, 0.1, 0.1])
cmap = matplotlib.cm.get_cmap("inferno")
# cmap = matplotlib.cm.get_cmap('plasma')
cls = []
# cl_ind = np.linspace(0,1,len(Pgrid))
# cl_ind = np.linspace(0.15,0.85,len(Pgrid))
cl_ind = np.array([0.15, 0.45, 0.7])
for j in range(len(Pgrid)):
    cls.append(cmap(cl_ind[j]))
print(cls)
# cls = np.array([ '0.3', 'r', 'magenta', 'Orange'])
NPgr = len(Pgrid)
NScl = "LightCyan"


cls_fill = [(0.9, 1.0, 1.0, 1.0)]
for j in np.arange(NPgr):
    clcur = cls[j]
    h, s, v = cl.rgb_to_hsv(clcur[0], clcur[1], clcur[2])
    s -= 0.7  # 0.3
    v += 0.6  # 0.2
    if s < 0:
        s = 0
    if v > 1:
        v = 1
    r, g, b = cl.hsv_to_rgb(h, s, v)
    cls_fill.append((r, g, b, 1.0))
cmap_fill = LinearSegmentedColormap.from_list("cmap_fill", cls_fill, N=4)


for i in np.arange(2, -1, -1):
    for j in range(5):
        savename = dirn + "P_%02d_%02d_n%d" % (i, j, nside)
        FLs_all, V, edges = get_field_lines(savename, gr.nline, Pgrid)
        rotph0 = 0
        fig = plt.figure(figsize=(4, cap_coeff_plot * 4), dpi=150)
        ax = plt.subplot(111)  # , title='Model of magnetic field of PSR J0030+0451')
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
        plt.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.1, hspace=0.1
        )

        lines = initialize_field_line_collection(NPgr, lws, cls)
        NS = plt.Circle((0, 0), 1, color=NScl, zorder=Nlayers / 2)
        ax.add_artist(NS)

        for k in range(NPgr):
            # FLs_cur = FLs_all[j]
            lines_cur = lines[k]
            for lc in lines_cur:
                ax.add_collection(lc)

        ax.set_xlim(-XZview, XZview)
        ax.set_ylim(-cap_coeff_plot * XZview, cap_coeff_plot * XZview)
        ax.set_xticks([])
        ax.set_yticks([])
        for rotph0 in [0, 15, 20]:  ##np.arange(0,360, 30):
            for incl in [-5]:
                fill_lines(FLs_all, NPgr, rotph0, incl, lines)

                plt.savefig(
                    plotdir
                    + "Frame_%d_%d_%02d_%03d_v7_poster.png"
                    % (i, j, np.abs(incl), rotph0)
                )

                # plt.show()
                # exit(1)

        plt.clf()
        ysize = 200
        xsize = 200
        theta = np.linspace(0, np.pi, ysize)
        phi = np.linspace(0, 2 * np.pi, xsize)
        longitude = np.radians(np.linspace(0, 360, xsize))
        latitude = np.radians(np.linspace(0, 180, ysize))
        PHI, THETA = np.meshgrid(phi, theta)
        grid_pix = hp.pixelfunc.ang2pix(nside, THETA, PHI)
        V[np.isnan(V)] = 0
        grid_map = V[grid_pix]
        grid_map = grid_map[:, ::-1]

        for rotph0 in [0]:  # np.arange(0,360,15):

            plt.figure(figsize=(4, 2))
            ax = plt.axes(projection=ccrs.Robinson())
            ax.set_global()
            ax.gridlines()
            x = np.rad2deg(longitude) + 90
            y = np.rad2deg(latitude) - 90
            ax.pcolormesh(
                x,
                y[::-1],
                grid_map,
                rasterized=True,
                cmap=cmap_fill,
                transform=ccrs.PlateCarree(),
            )
            ax.contour(
                x,
                y[::-1],
                grid_map,
                levels=np.unique(V),
                colors=cls,
                linewidths=2.5,
                transform=ccrs.PlateCarree(),
            )
            # plt.tight_layout()

            plt.subplots_adjust(
                left=0, right=1, bottom=0, top=1, wspace=0.1, hspace=0.1
            )
            plt.savefig(plotdir + "Map_%d_%d_%03d_v7_poster.png" % (i, j, rotph0))
            # plt.show()
