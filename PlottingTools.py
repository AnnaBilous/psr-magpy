import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R
import matplotlib

def get_Nlayers(XZview, YtoX, K):
    Yview = XZview * YtoX
    Nlayers = int(2 * Yview * K)
    return Nlayers


def initialize_subplot(XZview, ZtoX):
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(-XZview, XZview)
    ax.set_ylim(-ZtoX * XZview, ZtoX * XZview)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return ax


def initialize_lines(ax, linewidth_func, zorder_layer, Nlayers):
    cur_line = []
    for k in zorder_layer:
        line_layer = LineCollection(
            np.empty((0, 2, 2)),
            #color=linecolor_func(k, Nlayers),
            zorder=zorder_layer[k],
            linewidths=linewidth_func(k, Nlayers),
            antialiaseds=1,
            alpha=0.8,
            capstyle="round",
            rasterized=True,
        )
        ax.add_collection(line_layer)
        cur_line.append(line_layer)

    return cur_line


def get_line_projection(FLs, phi_rot, theta_rot):
    rotation_longitude = R.from_rotvec([np.deg2rad(phi_rot) * np.array([0, 0, 1])])
    rotation_latitude = R.from_rotvec([np.deg2rad(theta_rot) * np.array([1, 0, 0])])
    r = rotation_latitude * rotation_longitude

    FLs_proj = [r.apply(FL).T for FL in FLs]

    return FLs_proj

def prepare_lines_for_plotting(FLs, phi_rot, theta_rot, XZview, YtoX, K):
    # Project all field lines
    FLs_proj = get_line_projection(FLs, phi_rot, theta_rot)
    
    segments_list = []
    y_all = np.array([])
        
    for Xproj, Yproj, Zproj in FLs_proj:
        # Compute segments and midpoints
        points = np.stack([Xproj, Zproj], axis=1).reshape(-1, 1, 2)
        segments_list.append(np.concatenate([points[:-1], points[1:]], axis=1))
        y_all = np.append(y_all, 0.5 * (Yproj[1:] + Yproj[:-1]))
  
    # Combine segments and compute z-order
    segments_all = np.vstack(segments_list)
    zorder_all = ((y_all + XZview * YtoX) * K).astype(int)
    
    return segments_all, zorder_all


def compute_fraction_of_line_length(FLs):
    l_all = np.array([])
 
    for FL in FLs:
 
        X,Y,Z = FL.T
        R = (X[:2]**2 + Y[:2]**2 + Z[:2]**2)**0.5
        dR = R[1] - R[0]
        if dR > 0:
            l_min = 0
            l_max = 1
        else:
            l_min = 1
            l_max = 0
        # Compute segments and midpoints
        l_all = np.append(l_all, np.linspace(l_min, l_max, len(Y[:-1]), endpoint=True))

    return l_all


def fill_lines(lines, segments_all, zorder_all, color_param_all, zorder_layer, colormap):
    colors =  colormap(color_param_all)
    
    # Update line collection data
    for k in zorder_layer:
        mask = zorder_all == k
        lines[k].set_segments(segments_all[mask])
        lines[k].set_edgecolor(colors[mask])

    return lines
