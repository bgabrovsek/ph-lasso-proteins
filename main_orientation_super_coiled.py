"""
Find good examples of lassos with specific topology as a demonstration for the paper
"""



from pathlib import Path
import os
import gzip
import pickle
from itertools import product
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA


from filters import find_matchings, find_maxima_with_ranges, smooth_and_threshold, smoothen_and_find_peaks
import matplotlib.pyplot as plt
from lasso import LassoExtractor, get_lasso
from ph import ph_extended_diagrams, bottleneck_dist
from settings import *
from plot import plot_diagrams, plot_3D_lasso, interactive_ph_plot

def plot_3d_points(loop, tail, special_loop_points=None, special_tail_points=None):
    """
    Plots the original and interpolated points in 3D.

    Parameters:
        A (numpy.ndarray): The original Nx3 array of points.
        interpolated_A (numpy.ndarray): The interpolated 3Nx3 array.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color='k', alpha=0.5, linestyle="-", marker="o", label='loop')


    if special_loop_points:
        print(special_loop_points)
        special_loop_points = np.array(special_loop_points)
        ax.plot(special_loop_points[:, 0], special_loop_points[:, 1], special_loop_points[:, 2], color='r', alpha=1.0,  marker="x", label='closest loop')


    # Plot interpolated points
    ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], color='b', alpha=0.5, linestyle="-", marker="o", label='tail')


    if special_tail_points:
        print(special_tail_points)
        special_tail_points = np.array(special_tail_points)
        ax.plot(special_tail_points[:, 0], special_tail_points[:, 1], special_tail_points[:, 2], color='c', alpha=1.0,  marker="x", label='closest loop')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()



def closest_point(points, P):
    """
    Finds the closest point in 'points' to point 'P' using Euclidean distance.

    Parameters:
    - points: (N, 3) NumPy array of 3D coordinates.
    - P: (3,) NumPy array representing the reference point (x, y, z).

    Returns:
    - closest_idx: Index of the closest point in 'points'.
    - closest_point: The closest point as a NumPy array.
    - closest_distance: Euclidean distance to the closest point.
    """
    distances = np.linalg.norm(points - P, axis=1)  # Compute Euclidean distances
    closest_idx = np.argmin(distances)  # Find the index of the smallest distance
    return closest_idx, points[closest_idx]


def compute_principal_vector(path_points):
    """
    Computes the principal direction of a 3D path using PCA.

    Parameters:
    - path_points: (N, 3) NumPy array of 3D coordinates representing the path.

    Returns:
    - principal_vector: (3,) NumPy array, unit vector of the principal direction.
    """
    path_points = np.array(path_points)
    pca = PCA(n_components=3)
    pca.fit(path_points)  # Fit PCA on path points
    principal_vector = pca.components_[0]  # First principal component (dominant direction)

    average = (path_points[-1] + path_points[0]) // 4 + (path_points[-2] + path_points[1]) // 4
    if np.dot(average, principal_vector) < 0:
        principal_vector = -principal_vector


    return principal_vector / np.linalg.norm(principal_vector)  # Normalize to unit vector


def mixed(a,b,c, normalize=False):
    if normalize:
        return np.dot(a/np.linalg.norm(a), np.cross(b/np.linalg.norm(b), c/np.linalg.norm(c)))
    else:
        return np.dot(a, np.cross(b, c))


def compute_orientation(
        loop_xyz, tail_xyz, deep_indices
):

    # Computes orientation of intersections, so we can figure out if they are supercoiled or not (based on normals)

    tail_points = [tail_xyz[m] for m in deep_indices]
    closest_loop_points = [closest_point(loop_xyz, tail_xyz[m])[1] for m in maxima]
    closest_loop_indices = [closest_point(loop_xyz, tail_xyz[m])[0] for m in maxima]

    signs = []
    for L, T in zip(closest_loop_indices, maxima):
        loop_segment = loop_xyz[L - 2:L + 3]
        tail_segment = tail_xyz[T - 2:T + 3]
        loop_s = compute_principal_vector(loop_segment)
        tail_s = compute_principal_vector(tail_segment)
        connecting_segment = tail_xyz[T] - loop_xyz[L]
        #print(loop_s.shape, connecting_segment.shape, tail_s.shape)
        print("mixed product:", m := mixed(loop_s, connecting_segment, tail_s), "normalized:", mixed(loop_s, connecting_segment, tail_s, normalize=True))
        signs.append(1 if m > 0 else (-1 if m < 0 else 0))
    return signs

if __name__ == "__main__":

    filename = "2MGS_A-13-39.pkl.gz"
    pdb = "2MGS"

    # filename = "1Y7W_A-31-221.pkl.gz"
    # pdb = "1Y7W"
    lasso = get_lasso("2MGS", "A", 0)

    print("Lasso selected:")
    print(lasso["pdb"], lasso["id"], lasso["chain"], lasso["bridge"])
    print("keys", lasso.keys())

    print("Deep (N)", lasso["deep_n"], "Shallow (N)", lasso["shallow_n"], "Deep (C)", lasso["deep_c"], "Shallow (C)", lasso["shallow_c"])

    """
    Only input needed for analysis:
    lasso["xyz"]["loop"] - xyz of loop (all atoms on backbone)
    lasso["xyz"]["c"] - xyz of c-tail
    lasso["xyz"]["n"] - xyz of n-tail
    lasso["deep_c"] - indices of deep c-tail atoms
    lasso["deep_n"] - indices of deep n-tail atoms
    """
    # ONLY INPUT NEEDED FOR ANALYSIS:
    # xyz of loop, lasso["xyz"]["loop"]
    # xyz of c-tail
    # xyz of n-tail
    # theoretical


    ph_diagrams_c = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"], use_cache=True)
    ph_diagrams_n = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["n"], use_cache=True)

    #print(ph_diagrams_c[0].shape, len(ph_diagrams_c[1]), "and", ph_diagrams_n[0].shape, len(ph_diagrams_n[1]))

    f_bottle_c = bottleneck_dist(ph_diagrams_c, BOTTLENECK_MULT_THRESHOLD_PRE)
    f_bottle_n = bottleneck_dist(ph_diagrams_n, BOTTLENECK_MULT_THRESHOLD_PRE)

    tailN, loop, tailC = lasso["xyz"]["n"], lasso["xyz"]["loop"], lasso["xyz"]["c"],
    f_smooth, peaks = smoothen_and_find_peaks(f_bottle_c, FILTER_WINDOW_SIZE, FILTER_MULT_THRESHOLD,
                                              BOTTLENECK_ABS_THRESHOLD_POST)

    maxima, maxima_ranges = peaks
    print("Found two PH peaks (intersections):")
    for p, r in zip(maxima, maxima_ranges):
        print("    Peak:", p, "Range:", r)
    print("Real intersections:", lasso["deep_c"])

    #interactive_ph_plot(lasso, ph_diagrams_c, "c", f_bottle_c, f_smooth, peaks)

    # Display plot if it makes sense
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # intersection_points = [tailC[m] for m in maxima]
    # plot_3D_lasso(tailN, loop, tailC, deep_xyz=intersection_points, terminus="C", current_atom_index=None, ax=ax)
    # plt.show()


    compute_orientation(loop_xyz=loop, tail_xyz=tailC, deep_indices=lasso["deep_c"])

    exit()


    # compute_orientation(bottle=data[key_bottle],
    #         deep=data["data"][key_deep],
    #         shallow=data["data"][key_shallow],
    #         threhold_abs=ta, window=w, threshold_rel=tr,
    #         atom_distance=d,
    #         ignore_non_lassos=True,
    #         png_filename=figures_path / filename.replace("-orientation.pkl.gz",".png"),
    #                     loop_xyz = loop,
    #                     tail_xyz = tailC)

