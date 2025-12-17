""""
Utils for computing persistent homology.
"""
import os
import pickle
import gzip
import hashlib

from pathlib import Path

import numpy as np

from tqdm import tqdm
from ripser import ripser
#from scipy.spatial.distance import cdist
from teaspoon.TDA.Distance import bottleneckDist

PH_CACHE_DIR = Path("/Users/bostjan/Dropbox/Code/Lasso_PH/ph_cache")

def remove_nans(points: np.ndarray):
    while any(np.isnan(points[0])):
        points = points[1:]
    while any(np.isnan(points[-1])):
        points = points[:-1]
    return points

def persistence_homology(points: np.ndarray, dimension=1, use_cache=False):
    """Compute PH diagram."""

    points = remove_nans(points)

    if use_cache:
        if (cached_result := _from_cache_ph_extended_diagrams(points, None)) is not None:
            return cached_result

    #print("PH points", points)
    result = ripser(points, maxdim=1)["dgms"][dimension]  # only interested in diagrams in homology dimension 1

    #print("ripser", ripser(points, maxdim=1))
    #print("PH result", result)
    #exit()

    if use_cache:
        _save_cache_ph_extended_diagrams(points, None, result)

    return result

def ph_extended_diagrams(points: np.ndarray, extension: np.ndarray, show_progress=True, use_cache=True):
    """ Compute PH diagrams of points u {x}, where x is from the extension."""
    base_result = []
    extended_results = []

    # compute base PH
    base_result = persistence_homology(remove_nans(np.array(points)))

    # first try to use cache
    if use_cache:
        if (cached_result := _from_cache_ph_extended_diagrams(points, extension)) is not None:
            return cached_result

    # join points + point from the extension
    for x in (tqdm(extension, desc="PH") if show_progress else extension):
        #print("x", end="")
        xyz = np.concatenate((points, np.array([x])), axis=0) if x is not None else np.array(points)
        xyz = remove_nans(xyz)
        extended_results.append(persistence_homology(xyz))

    #print()
    result = (base_result, extended_results)

    _save_cache_ph_extended_diagrams(points, extension, result)

    return result


def _from_cache_ph_extended_diagrams(points: np.ndarray, extension: np.ndarray, force_recompute=False):
    """Computes persistence homologies of the lasso containing the main loop + one atom from the tail.
    For each atom in the tail, it computes the PH
    Args:
        loop: numpy array of xyz coordinates of atoms in the loop (dimension (N, 3))
        tail: numpy array of xyz coordinates of atoms in the tail (dimension (M, 3))
        show_progress: True to show progress bar
        dimensions: what dimension of homology groups to compute, returns "dgms"  if None
    Returns:
        list of persistence homologies, one for each atom in the tail
    """
    if force_recompute:
        return None

    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")

    if extension is not None and not isinstance(extension, np.ndarray):
        raise TypeError("extension points must be a numpy array or None")

    points_hash = hashlib.sha256(points.tobytes()).hexdigest()
    extension_hash = hashlib.sha256(extension.tobytes()).hexdigest() if extension is not None else ""
    h = points_hash + extension_hash  # full hash of loop + tail
    file_path = PH_CACHE_DIR / (h + ".gz")  # the file name is the hash of both the loop and the tail
    if not os.path.exists(file_path):
        return None
    else:
        with gzip.open(file_path, 'rb') as f:
            #print("loading")
            return pickle.load(f)




def _save_cache_ph_extended_diagrams(points: np.ndarray, extension: np.ndarray, ph_diagrams: tuple):
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")

    if extension is not None and not isinstance(extension, np.ndarray):
        raise TypeError("extension points must be a numpy array or None")

    points_hash = hashlib.sha256(points.tobytes()).hexdigest()
    extension_hash = hashlib.sha256(extension.tobytes()).hexdigest() if extension is not None else ""
    h = points_hash + extension_hash  # full hash of loop + tail
    file_path = PH_CACHE_DIR / (h + ".gz")  # the file name is the hash of both the loop and the tail

    with gzip.open(file_path, 'wb') as f:
        pickle.dump(ph_diagrams, f)



if __name__ == "__main__":
    # --- Generate 100 points on a unit circle in 3D (z=0)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle_points = np.vstack([
        np.cos(theta),
        np.sin(theta),
        np.zeros_like(theta)
    ]).T  # shape (100, 3)

    points = circle_points

    # --- Generate extension points along the z-axis (perpendicular line)
    z_values = np.linspace(-1.0, 1.0, 200)  # 5 sample points along the line
    extension = np.vstack([
        np.zeros_like(z_values),
        np.zeros_like(z_values),
        z_values
    ]).T  # shape (5, 3)

    # Compute PH(A)
    from time import time
    t = time()
    base_ph = ph_extended_diagrams(points, extension)
    print(time()-t)

    t = time()
    base_ph = ph_extended_diagrams(points, extension)
    print(time()-t)


    # Compute PH(A ∪ {b}) for each b in extension
    extended = ph_extended_diagrams(points, extension, show_progress=False)
    print("Computed", len(extended), "PH diagrams.")





#
def bottleneck_dist(diagram_pair, threshold_mult):
    """
    Computes the bottleneck distance-based cost function for persistence diagrams.

    This function applies a threshold to filter persistence pairs in the loop diagram
    and atom diagrams, then computes bottleneck distances relative to the loop diagram.

    Args:
        diagram_loop (np.ndarray): An (n,2) array representing the persistence
            diagram of the loop (without atoms), where each row contains (birth, death) pairs.
        diagrams_atoms (list of np.ndarray): A list of (n,2) arrays representing
            persistence diagrams of atoms in the tail.
        thresh_mult (float): A multiplier applied to the maximum lifespan in
            `diagram_loop`, determining the threshold for filtering short-lived features.

    Returns:
        np.ndarray: An array of bottleneck distances between the thresholded persistence
            diagrams of atoms and the thresholded loop persistence diagram.

    Notes:
        - The threshold is computed as `thresh_mult * max_lifespan`, where `max_lifespan`
          is the longest persistence interval in `diagram_loop`.
        - Only persistence pairs exceeding the threshold are retained.
        - Bottleneck distances are computed using the filtered diagrams."""

    diagram_loop, diagrams = diagram_pair

    if threshold_mult is not None:


        # compute threshold (what fraction of longest lifespans we will take)
        lifespan_threshold = np.max(diagram_loop[:, 1] - diagram_loop[:, 0]) * threshold_mult

        # only keep births/death above the threshold
        diagram_loop_filtered = diagram_loop[(diagram_loop[:, 1] - diagram_loop[:, 0]) >= lifespan_threshold]

        return np.array([bottleneckDist(diagram_loop_filtered,
                               diag[(diag[:, 1] - diag[:, 0]) >= lifespan_threshold],
                               matching=False,
                               plot=False)
                         for diag in diagrams
                         ])

    else:
        return np.array([bottleneckDist(diagram_loop,
                               diag,
                               matching=False,
                               plot=False)
                         for diag in diagrams
                         ])