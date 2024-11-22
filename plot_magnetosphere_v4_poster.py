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
blcl = "k"
grcl = "0.6"
mx = 6
Rns = 1  # Radius of neutron star
RLC = 10  # Radius of the light cylinder
K = 10  # Number of zorder layers per unit depth
XZview = RLC  # XZ viewport size
Nlayers = int(RLC * 2 * K)  # Total number of zorder layers
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
lw_max = 0.4
lw_med = 0.1

pli = -np.log(lw_med - lw_min) / (lw_max - lw_min) / np.log(2)


def lw_func(layer_no):
    return (lw_max - lw_min) * (layer_no * 1.0 / Nlayers) ** pli + lw_min


def get_edges(cr, linenum, nside, inner):
    # nnb = hp.pixelfunc.get_all_neighbours(nside, cr)
    cr_not_inner = np.setdiff1d(cr, inner)
    Ncr = len(cr)
    Ncr_not_inner = len(cr_not_inner)
    is_edge = np.zeros(Ncr_not_inner, dtype=bool)
    nnb = hp.pixelfunc.get_all_neighbours(nside, cr_not_inner)
    for ie in range(Ncr_not_inner):
        nnb_to_comp = np.copy(nnb[:, ie])
        nnb_to_comp = nnb_to_comp[nnb_to_comp > 0]
        # is_edge[ie] = ~(np.all( np.isin(nnb_to_comp, cr) ))
        is_edge[ie] = len(nnb_to_comp[np.isin(nnb_to_comp, cr)]) < (
            len(nnb_to_comp) - 2
        )

    ln_cr = cr_not_inner[is_edge]
    return ln_cr


def trim_line(r_fl, RLC):
    cyl = (r_fl[0, :] ** 2 + r_fl[1, :] ** 2) ** 0.5
    z = np.abs(r_fl[2, :])
    # cr = (cyl<RLC)&(z<RLC)
    # r_fl = r_fl[:,cr]
    k = 1.01
    nstep = np.arange(len(r_fl[0, :]))
    if np.max(cyl) > (RLC * k):
        LCp = nstep[cyl > (RLC * k)]
        r_fl = r_fl[:, : LCp[0]]
        """
            #print("x",LCp)
            if len(LCp)>1:
             A1 = np.max(cyl[:LCp[0]])
             A2 = np.max(cyl[LCp[-1]:])
             if A1>A2: 
              LCp = LCp[0]
              r_fl = r_fl[:,:LCp]        
             if A1<=A2: 
              LCp = LCp[-1]
              r_fl = r_fl[:,LCp:]        
            """
    return r_fl


def initialize_field_line_collection(NPgr, lws, cls):
    # XZview = 10
    norm = plt.Normalize(
        -XZview, XZview
    )  # Normalizing zorder colors to the size of XZ viewport. Can be tweaked for aesthetics.
    # zorder_layer = zorder

    # Setting line widths at specific zorder
    # y_layer = zorder_layer*1./K-RLC # recovering y-coordinate of particular zorder layer

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
    Rvb = np.load(savename + "_FLstat.npy")
    # Rvb = np.load(savename)
    ind = Rvb[:, 0].astype(int)
    # plt.plot(ind)
    # plt.show()
    cylmax = Rvb[:, 1]
    radmax = Rvb[:, 2]
    zmax = Rvb[:, 3]
    TH = gr.th[ind]
    PH = gr.ph[ind]

    NPgr = len(Pgrid)
    rlc_grid = 4.77 * Pgrid

    TH_all = np.copy(gr.th)
    PH_all = np.copy(gr.ph)
    V = np.zeros(nline)
    V_edge = np.zeros((NPgr, nline), dtype=int)
    ###plt.plot(PH_all, TH_all, 'k.')

    for rr in range(NPgr):
        # cap_coeff = (5**0.5)/2.
        # cap_coeff = (2**0.5)
        cap_coeff = (3**1.5) / 2.0
        cr = (cylmax > rlc_grid[rr]) | (
            (cylmax < rlc_grid[rr]) & (zmax > (cap_coeff * rlc_grid[rr]))
        )
        # print(zmax[hcr])

        # exit(1)
        V[ind[cr]] = rr + 1
        # plt.scatter(PH, TH, marker='o', c = radmax)
        # plt.colorbar()
        # plt.scatter(PH[cr], TH[cr], marker='.', color='k')
        # plt.show()
        # plt.plot(PH[cr], cylmax[cr], '.')
    # plt.show()
    # exit(1)
    V[V == 0] = np.NaN
    # hp.visufunc.cartview(V, fig=1, title='', notext=True, cbar=False,  margins=(0,0,0,0), cmap='viridis', badcolor='0.9', max=4, coord='C')
    # plt.show()

    # plt.plot(PH, TH, '.')
    # plt.show()
    linenum = np.arange(nline)
    nside = hp.pixelfunc.npix2nside(nline)

    ###edge_weird = np.loadtxt("edge.txt", dtype=int)
    ###plt.figure()
    ###plt.plot(PH_all[edge_weird], TH_all[edge_weird], 'ro')
    edges = []

    for i in range(NPgr):
        subfl = linenum[V >= (i + 1)]
        if i < (NPgr - 1):
            inner = linenum[V >= (i + 2)]
        else:
            inner = []
        edge = get_edges(subfl, linenum, nside, inner=inner)
        # plt.plot(PH_all[edge_weird], TH_all[edge_weird], 'ro')
        ###plt.plot(PH_all[edge], TH_all[edge], '.')
        # for jj in edge:
        # plt.scatter([PH_all[jj]], [TH_all[jj]], c=cylmax[ind==jj], marker='.')
        # plt.colorbar()
        # plt.show()
        # plt.suptitle("%.2f %.2f"%(rlc_grid[i], rlc_grid[i]*cap_coeff))
        # plt.plot(PH_all[edge], TH_all[edge], 'k.')
        """
     if i==1:
      for jj in edge:
       if np.squeeze(cylmax[ind==jj])>11:   
        plt.plot([PH_all[jj]], [TH_all[jj]], 'r.') #cylmax[ind==jj], 'r.')

     plt.show()
     """
        edges.append(edge)
    ###plt.show()

    sol_ind = np.array([])
    with open(savename + ".pickle", "rb") as tp:
        # with open("../../Run_03_FL/R_01_05.pickle", 'rb') as tp:
        sols = pickle.load(tp)
        Nsols, _ = np.shape(sols)
        sol_ind = np.zeros(Nsols)
        sol_ind = np.array(list(zip(*sols))[0])
        FLs_all = []
        for j in range(NPgr):
            edge = edges[j]
            FLs = []
            for k in range(len(edge)):
                ind = np.where(sol_ind == edge[k])[0]
                if len(ind) > 0:
                    # print(ind)
                    sol = sols[ind[0]][1]
                    t = np.array([])
                    # print(len(sol.t))
                    for l in range(len(sol.t) - 1):
                        t = np.append(t, np.linspace(sol.t[l], sol.t[l + 1], 10))
                    r_fl = sol.sol(t)
                    """
         if j==0:
          cm = np.max((r_fl[0,:]**2+r_fl[1,:]**2)**0.5)

          plt.plot(r_fl[0,:], r_fl[2,:], 'k', lw=0.1)
          if cm>13:
           #print(sol)   
           plt.plot(r_fl[0,:20], r_fl[2,:20], 'r')
           print(edge[k])
           #print(r_fl[:,0])
          """
                    # print(cylmax[ind==edge[k]])

                    # print(len(r_fl[0,:]))
                    r_fl = trim_line(r_fl, XZview)
                    # print(len(r_fl[0,:]),'---')
                    FLs.append(r_fl)
                # print(np.shape(sol.sol(t)))
                # plt.plot(r_fl[0,:], r_fl[2,:])
            # plt.show()
            FLs_all.append(FLs)
    # plt.show()
    """
     for i in lev:
        r_fl = np.copy(lines[:,  ls[i]:ls[i+1]])
        #r_fl = trim_line(i, ls, lines, cylmax, zmax, 1.2)
        nlev[i] = len(r_fl[0,:])
        #ax.plot( r_fl[0,:], r_fl[1,:], r_fl[2,:],   color='b', lw=0.05)
    """

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
        segments = np.empty((0, 2, 2))
        y = np.array([])
        Nline_cur = len(FLs)
        for k in range(Nline_cur):
            r_fl = FLs[k]
            X = r_fl[0, :]
            Y = r_fl[1, :]
            Z = r_fl[2, :]
            # plt.plot(Y, Z, 'k')
            Xproj, Yproj, Zproj = (r.apply(np.transpose((X, Y, Z)))).T
            points = np.array([Xproj, Zproj]).T.reshape(-1, 1, 2)
            segments_cur = np.concatenate([points[:-1], points[1:]], axis=1)
            segments = np.append(segments, segments_cur, axis=0)
            y = np.append(y, 0.5 * (Yproj[1:] + Yproj[:-1]))
            zorder = ((y + RLC) * K).astype(int)  # zorder from 0 to Nlayers
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
for j in np.arange(3):
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


for i in [2]:  # np.arange(2,-1,-1):
    for j in [0]:  # range(5):
        savename = dirn + "P_%02d_%02d_n%d" % (i, j, nside)
        FLs_all, V, edges = get_field_lines(savename, gr.nline, Pgrid)
        # np.save("Vplay_aligned.npy", V)
        # exit(1)
        # V = np.load("Vplay.npy")
        rotph0 = 0
        fig = plt.figure(figsize=(6, 6), dpi=150)
        # ax = plt.subplot2grid((3,1), (0,0), rowspan=2, frame_on=True) #, title='Model of magnetic field of PSR J0030+0451')
        ax = plt.subplot(111)  # , title='Model of magnetic field of PSR J0030+0451')
        plt.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1
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
        ax.set_ylim(-XZview, XZview)
        ax.set_xticks([])
        ax.set_yticks([])
        for rotph0 in [0, 15]:  ##np.arange(0,360, 30):
            for incl in [0, -10]:
                fill_lines(FLs_all, NPgr, rotph0, incl, lines)
                # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)

                plt.savefig(
                    plotdir
                    + "Frame_%d_%d_%02d_%03d_v6_poster.png"
                    % (i, j, np.abs(incl), rotph0)
                )
                # plt.show()

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

            plt.figure(figsize=(6, 3))
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
            plt.savefig(plotdir + "Map_%d_%d_%03d_v6_poster.png" % (i, j, rotph0))
            # plt.show()
