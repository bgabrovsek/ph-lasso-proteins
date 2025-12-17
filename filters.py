import numpy as np


from scipy.signal import find_peaks

def smoothen_and_find_peaks(data, window_size, threshold_mult_filter, threshold_absolute):

    data_filtered = smooth_and_threshold(data=data, window=window_size, threshold_mult=threshold_mult_filter, threshold_absolute=threshold_absolute)  # 4 and 0.05 at the beginning
    maxima_ranges_pair = find_maxima_with_ranges(data=data_filtered)
    return data_filtered, maxima_ranges_pair


def smooth(data, window=3):
    """ smoothens data using a window 11..1."""
    return np.convolve(data, np.ones(window)/window, mode='same')

def cut_off_peaks(values, threshold):
    """Smooth out peaks by limiting changes beyond a threshold."""
    values = np.array(values)  # Convert to NumPy array for easy manipulation
    smoothed = values.copy()   # Copy original list

    for i in range(1, len(values)):
        if abs(values[i] - smoothed[i-1]) < threshold:  # Only modify large jumps
            smoothed[i] = smoothed[i-1]  # Create a plateau

    return smoothed

def count_peaks(data, threshold):
    # count how many times the data reaches the threshold
    data = np.concatenate(([0], data, [0]))  # pad with zeroes
    above_threshold = data > threshold  # Boolean array: True where values are above threshold
    crossings = np.diff(np.concatenate(([False], above_threshold)))  # Detect start of above-threshold regions
    return np.sum(crossings == True)//2  # Count only rising edges (False -> True transitions)

#
# def smooth_and_threshold(data, window, threshold_mult):
#     """    Smooths a list of values by limiting abrupt changes beyond a given threshold.
#
#     This function iterates through the list and ensures that the difference between
#     consecutive values does not exceed the specified threshold. If a jump larger
#     than the threshold is detected, the value is replaced with the previous value,
#     effectively creating a plateau.
#
#     Args:
#         data:
#         window:
#         threshold_mult:
#
#     Returns:
#
#     """
#     if (window == 1 and threshold_mult == 0) or len(data) == 0:
#         return data
#
#     threshold = (np.max(data) - np.min(data)) * threshold_mult
#
#     data[np.abs(data) < 0.1] = 0  # if it is veeeery small, it will put it to zero
#
#     # cut off peaks
#     data = cut_off_peaks(data, threshold)
#
#     # smoothen the data
#     if window > 1:
#         data = np.convolve(data, np.ones(window)/window, mode='same')
#
#     data[data < 0.1] = 0  # if it is veeeery small, it will put it to zero
#
#     # cut off peaks
#     data = cut_off_peaks(data, threshold)
#
#     data[data < 0.1] = 0  # if it is veeeery small, it will put it to zero
#     # Here Paolo put the limit threshold, not 0.1, is it a mistake?
#
#     return data


def smooth_and_threshold(data, window, threshold_mult, threshold_absolute):
    """    Smooths a list of values by limiting abrupt changes beyond a given threshold.

    This function iterates through the list and ensures that the difference between
    consecutive values does not exceed the specified threshold. If a jump larger
    than the threshold is detected, the value is replaced with the previous value,
    effectively creating a plateau.

    Args:
        data:
        window:
        threshold_mult:

    Returns:

    """

    threshold = (np.max(data) - np.min(data)) * threshold_mult

    data[np.abs(data) < threshold_absolute] = 0  # if it is veeeery small, it will put it to zero

    # cut off peaks
    data = cut_off_peaks(data, threshold)

    # smoothen the data
    if window > 1:
        data = np.convolve(data, np.ones(window)/window, mode='same')

    data[data < threshold_absolute] = 0  # if it is veeeery small, it will put it to zero

    # cut off peaks
    data = cut_off_peaks(data, threshold)

    data[data < threshold_absolute] = 0  # if it is veeeery small, it will put it to zero
    # Here Paolo put the limit threshold, not 0.1, is it a mistake?

    return data


#
#
# def find_max_min_smooth(data, window, threshold_mult):
#     """finds the local minimums and maxiums of data, after it is smoothened by a window of size window and using a
#     (relaitive) threshold.
#
#     Args:
#         data: 1D aerray
#         window: smoothing windows size
#         threshold_mult:
#
#     Returns: a tuple of:
#     - indexes of maximums,
#     - indexes of minimums,
#     - interval of maximums in case the function is horizontal at the maximum
#     - interval of minimums in case the function is horizontal at the minimum
#     """
#
#     if len(data) <= 2:
#         return [], [], [], []
#
#     data = smooth_and_threshold(data, window, threshold_mult)
#     first_derivative = np.gradient(data)
#
#     max_f = []
#     min_f = []
#     max_f_intervals = []
#     min_f_intervals = []
#     #now for the plateau minima and maxima
#     #beginning data: the initial piece is going up
#     first_derivative[0] = 1
#     first_derivative[-1] = 1
#     plateau_st = 1
#     plateau_end = 0
#     for point in range(len(data)-1):
#         if first_derivative[point] == 0:
#             #print("I start the piece here")
#             if plateau_st == -1:
#                 plateau_st = point
#         elif plateau_st != -1:
#             #print("It ends here, and I add ...")
#             plateau_end = point
#             if first_derivative[plateau_st - 1] > 0:
#                 if first_derivative[plateau_end] <0:
#                     #print("A maxima")
#                     max_f.append((plateau_end + plateau_st)//2)
#                     max_f_intervals.append((plateau_st, plateau_end))
#             if first_derivative[plateau_st - 1] < 0:
#                 if first_derivative[plateau_end] > 0:
#                     #print("A minima")
#                     min_f.append((plateau_end + plateau_st)//2)
#                     min_f_intervals.append((plateau_st, plateau_end))
#
#             plateau_st = -1
#
#     #now for the normal maxima and minima
#     for point in range(len(data)-1):
#         if first_derivative[point] > 0:
#             if first_derivative[point + 1] < 0:
#                 max_f.append(point)
#                 max_f_intervals.append((point, point))
#         if first_derivative[point] < 0:
#             if first_derivative[point + 1] > 0:
#                 min_f.append(point)
#                 min_f_intervals.append((point, point))
#
#     return max_f, min_f, max_f_intervals, min_f_intervals
#

# def cost_rank_dist(diagrams, rank=0):
#     """ gets the rank's homology (rank=0 is biggest, rank=1 is 2nd biggest"""
#     """Returns an array, which is the criteria of how to decide if a point is intersecting the loop.
#     The function returns a list of the maximal living time (death - birth of homology) for 1-st homology.
#     Args:
#         phs: a list of ripser diagrams, one for each atom in the tail
#     Returns:
#         max distances of 1-th homologies in diagrams
#     """
#     result = []
#     homology = 1
#     lifetimes = [d[:,1] - d[:,0] for d in diagrams]  # lifetimes of homology generators
#     #max_lifetime = [np.max(lt[lt != np.inf]) for lt in lifetimes]
#     max_lifetime = [sorted(lt[lt != np.inf], reverse=True)[rank] for lt in lifetimes]
#     return np.array(max_lifetime)

# def dist_w(diagrams):
#     distances = []
#     d0 = diagrams[0]
#     for d_ in diagrams:
#         d = d_
#         dist = persim.bottleneck(d0, d, matching=False)
#         distances.append(dist)
#     return distances




def find_maxima_with_ranges(data):
    """
    Finds local maxima and their corresponding start and end indices in a 1D NumPy array.

    Args:
        data (np.ndarray): 1D array of numerical data.
    Returns:
        list of max_values and list of tuples: Each tuple contains (start_index, end_index).

    Does not smooth
    """
    #data = np.array(data)  # Ensure it's a NumPy array
    #data = smooth_and_threshold(data, window, threshold_mult)
    peaks, _ = find_peaks(data)  # Find local maxima

    maxima_ranges = []
    maxima = []

    for peak in peaks:
        max_value = data[peak]

        # Move left to find start of plateau
        start = peak
        while start > 0 and data[start - 1] == max_value:
            start -= 1

        # Move right to find end of plateau
        end = peak
        while end < len(data) - 1 and data[end + 1] == max_value:
            end += 1

        maxima.append((end+start)//2)  #max_value)
        maxima_ranges.append((start, end))

    return maxima, maxima_ranges
#
# def find_maxima_with_ranges_smooth(data, window, threshold_mult):
#     """
#     Finds local maxima and their corresponding start and end indices in a 1D NumPy array.
#
#     Args:
#         data (np.ndarray): 1D array of numerical data.
#         window:
#         threshold_mult:
#     Returns:
#         list of max_values and list of tuples: Each tuple contains (start_index, end_index).
#     """
#     #data = np.array(data)  # Ensure it's a NumPy array
#     data = smooth_and_threshold(data, window, threshold_mult)
#     peaks, _ = find_peaks(data)  # Find local maxima
#
#     maxima_ranges = []
#     maxima = []
#
#     for peak in peaks:
#         max_value = data[peak]
#
#         # Move left to find start of plateau
#         start = peak
#         while start > 0 and data[start - 1] == max_value:
#             start -= 1
#
#         # Move right to find end of plateau
#         end = peak
#         while end < len(data) - 1 and data[end + 1] == max_value:
#             end += 1
#
#         maxima.append((end+start)//2)  #max_value)
#         maxima_ranges.append((start, end))
#
#     return maxima, maxima_ranges



def find_matchings(measured_data, ground_truth, distance):
    """
    Counts positive matchings, false positives, and false negatives between data and ground values.

    Parameters:
    -----------
    measured_data : list of tuples (float, float)
        A list of (lower_bound, upper_bound) pairs representing detected regions.
    ground_truth : list of float
        The ground truth values.
    distance : float
        The tolerance distance to extend bounds.

    Returns:
    --------
    tuple (int, int, int)
        - Positive matchings: Number of detected bounds that match a ground truth value.
        - False positives: Number of detected bounds that do not match any ground truth value.
        - False negatives: Number of ground truth values that are not covered by any detected bound.
    """

    # Convert ground to a set for efficient lookups
    ground_set = set(ground_truth)

    # Count positive matchings and false positives
    positive_matchings = 0
    false_positives = 0
    covered_ground = set()

    for lower_bound, upper_bound in measured_data:

        match_found = any((lower_bound - distance) <= g <= (upper_bound + distance) for g in ground_truth)

        if match_found:
            positive_matchings += 1
            covered_ground.update({g for g in ground_truth if (lower_bound - distance) <= g <= (upper_bound + distance)})
        else:
            false_positives += 1

    # False negatives: Ground truth values that are not covered by any detected bound
    false_negatives = len(ground_set - covered_ground)

    return positive_matchings, false_positives, false_negatives

def get_confusion_stats(bottle, deep, shallow, threhold_abs, window, threshold_rel, atom_distance, ignore_non_lassos):
    """ Analyse if the PH and minimal surface method are getting same results"""

    # do not consider
    if ignore_non_lassos and len(deep) == 0:
        return 0, 0, 0, 0

    # smoothen the data
    smooth_bottle = smooth_and_threshold(bottle, window=window, threshold_mult=threshold_rel, threshold_absolute=threhold_abs)
    maxima, maxima_ranges = find_maxima_with_ranges(smooth_bottle)
    tp, fp, fn = find_matchings(measured_data=maxima_ranges, ground_truth=deep, distance=atom_distance)
    tp_, fp_, fn_ = find_matchings(measured_data=maxima_ranges, ground_truth=sorted(deep + shallow), distance=atom_distance)

    # include shallow intersections only if they produce better results

    # if a shallow matching is found, count it in.
    all_intersections = len(deep)
    if tp_ > tp:
        all_intersections += (tp_ - tp)
        tp = tp_
    fp = min(fp, fp_)
    fn = min(fn, fn_)

    return tp, fp, fn, all_intersections


def compute_statistics_from_confusion(results_list):

    positives = 0
    false_negatives = 0
    false_positives = 0
    all_intersections = 0

    for tp, fp, fn, ai in results_list:
        positives += tp
        false_positives += fp
        false_negatives += fn
        all_intersections += ai

    precision = positives / (positives + false_positives) if positives + false_positives != 0 else 0
    recall = positives / (positives + false_negatives) if positives + false_negatives != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    intersection_quality = positives / all_intersections if all_intersections != 0 else 0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "intersection_quality": intersection_quality,
        "all_intersections": all_intersections,
        "positives": positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,

    }

    return f1, precision, recall, intersection_quality, all_intersections, positives, false_positives, false_negatives, ta, w, tr
