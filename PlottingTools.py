import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R


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
    return ax


def initialize_lines(ax, linewidth_func, linecolor_func, zorder_layer, Nlayers):
    cur_line = []
    for k in zorder_layer:
        line_layer = LineCollection(
            np.empty((0, 2, 2)),
            color=linecolor_func(k, Nlayers),
            zorder=zorder_layer[k],
            linewidths=linewidth_func(k, Nlayers),
            antialiaseds=1,
            alpha=1.0,
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


def fill_lines(FLs, phi_rot, theta_rot, lines, XZview, YtoX, K, zorder_layer):
    # Project all field lines
    FLs_proj = get_line_projection(FLs, phi_rot, theta_rot)

    segments_all, zorder_all = [], []

    segments_list, y_list = [], []

    for Xproj, Yproj, Zproj in FLs_proj:
        # Compute segments and midpoints
        points = np.stack([Xproj, Zproj], axis=1).reshape(-1, 1, 2)
        segments_list.append(np.concatenate([points[:-1], points[1:]], axis=1))
        y_list.append(0.5 * (Yproj[1:] + Yproj[:-1]))

    # Combine segments and compute z-order
    segments_all = np.vstack(segments_list)
    y = np.concatenate(y_list)
    zorder_all = ((y + XZview * YtoX) * K).astype(int)

    # Update line collection data
    for k in zorder_layer:
        mask = zorder_all == k
        lines[k].set_segments(segments_all[mask])
        lines[k].set_color("w")

    return lines
