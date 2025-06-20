from sys import float_info
import numpy as np
from numpy.fft import fft, ifft

EPS = float_info.epsilon

def mass_pre(data, window_size):
    n, dim = data.shape
    x_mat = np.zeros((2 * n, dim))
    x_mat[:n] = data
    X = np.fft.fft(x_mat, axis=0)
    cum_sumx2 = (data ** 2).cumsum(axis=0)
    sumx2 = cum_sumx2[window_size - 1 : n] - np.append(
        np.zeros((1, dim)), cum_sumx2[: n - window_size], axis=0
    )
    return X, sumx2

def mass(X, y, n, m, dim, sumx2, dist_func):
    """Calculate the distance profile using the MASS algorithm.

    - X: the data fft
    - y: the query 
    - n: the number of rows in the query
    - m: the sliding window size
    - dim: feature dimension
    - sumx2: the precomputed sum of squares

    returns (dist, z, sumy2)
    where
        dist: the distance profile
        z: the last product
        sumy2: the sum of squared query values
    """

    # computing dot product in O(n log n) time
    y_mat = np.zeros((2 * n, dim))
    y_mat[:m] = y[::-1]
    Y = np.fft.fft(y_mat, axis=0)
    Z = X * Y
    z = np.real(np.fft.ifft(Z, axis=0)[m - 1 : n])
    sumy2 = (y ** 2).sum(axis=0)
    dist = dist_func(sumx2, sumy2, z)
    return dist, z, sumy2

def euc_dist(sumx2, sumy2, z):
    return (sumx2 - 2 * z + sumy2).sum(axis=1)

def cosine_dist(sumx2, sumy2, z):
    # sumx2 shape = (# query windows, dim)
    # sumy2 shape = (dim, )
    z_sum = z.sum(axis = 1)
    magnitude_y = np.sqrt(sumy2.sum())
    magnitude_x = np.sqrt(sumx2.sum(axis=1))
    if magnitude_y == 0:
        if np.all(sumy2 == 0):
            return np.ones(len(magnitude_x))
        magnitude_y = EPS
    denom = magnitude_x * magnitude_y
    inv_denom = np.zeros_like(denom) 
    np.divide(1.0, denom, out=inv_denom, where=denom != 0)
    result = 1.0 - z_sum * inv_denom
    if np.any(result < 0):
         print(f"HERE: min_val {np.min(result)}")
    # return np.maximum(1.0 - z_sum * inv_denom, 0.0)
    return result

def partition_find_min_x(values, k):
    """Find the k smallest values in an array 

    Assumes k is an int >= 1
    
    Args:
        values (np.ndarray):
        k (int): An integer >= 1

    Returns:
        tuple[np.ndarray]: Sorted min k values (same dtype at values) and their indices in *values* (np.uint64)
        - If len(values) <= k, then a sorted copy of 'values' is returned (with indices)
    """ 
    n_values = len(values)
    if n_values <= k:
        indices = np.argsort(values)
        return np.take_along_axis(values, indices), indices
    
    indices = np.argpartition(values, k)[:k]
    indices = indices[np.argsort(values[indices])]
    return values[indices], indices

def z_score_norm(distance_profile, with_min_max = True):
    mean_d = np.mean(distance_profile)
    std_d = np.std(distance_profile)
    z_profile = (distance_profile - mean_d) / max(std_d, EPS)
    return z_profile if not with_min_max else min_max_norm(z_profile)

def min_max_norm(distance_profile):
    global_min = np.min(distance_profile)
    global_max = np.max(distance_profile)
    normalized_matrix = (distance_profile - global_min) / max((global_max - global_min), EPS)
    return normalized_matrix

def _weighted(prev_combined_profile, curr_distance_profile, curr_weight):
    prev_combined_profile += curr_weight*curr_distance_profile

def _post_none(weighted_distance_profile):
    return

def _geometric(prev_combined_profile, curr_distance_profile, curr_weight):
    prev_combined_profile += curr_weight * np.log(np.maximum(curr_distance_profile, EPS))
    
def _post_geometric(weighted_distance_profile):
    np.exp(weighted_distance_profile, out=weighted_distance_profile)

def _harmonic(prev_combined_profile, curr_distance_profile, curr_weight):
    elems = np.zeros_like(curr_distance_profile)
    np.divide(curr_weight, curr_distance_profile, out=elems, where=curr_distance_profile != 0)
    prev_combined_profile += elems

def _post_harmonic(weighted_distance_profile):
    np.divide(1.0, weighted_distance_profile, out=weighted_distance_profile, where= weighted_distance_profile != 0)

def _rms(prev_combined_profile, curr_distance_profile, curr_weight):
    prev_combined_profile += curr_weight * (curr_distance_profile ** 2)

def _post_rms(weighted_distance_profile):
    np.sqrt(weighted_distance_profile, out=weighted_distance_profile)

def _softmax(prev_combined_profile, curr_distance_profile, curr_weight):
    # print("")
    # print(curr_distance_profile)
    # print(np.exp(curr_weight*curr_distance_profile))
    prev_combined_profile += np.exp(curr_weight*curr_distance_profile)

def _post_softmax(weighted_distance_profile):
    weighted_distance_profile /= np.sum(weighted_distance_profile)

def get_score_combinator(combination_method):
    """returns two functions: during iteration func, after iteration func"""
    if combination_method == 'weighted':
        return _weighted, _post_none
    if combination_method == 'geometric':
        return _geometric, _post_geometric
    if combination_method == 'rms':
        return _rms, _post_rms
    if combination_method == 'softmax':
        return _softmax, _post_softmax
    if combination_method == 'harmonic':
        return _harmonic, _post_harmonic
    
    raise ValueError(f"Invalid combination type {combination_method}")
    

# def geometric_sum()
def multi_score_combination(prev_combined_profile, curr_distance_profile, combination_method, curr_weight):
    if combination_method == 'weighted' or 'softmax':
        if prev_combined_profile is None:
            return curr_weight*curr_distance_profile
        else:
           prev_combined_profile += curr_weight*curr_distance_profile
           return prev_combined_profile
    elif combination_method == 'geometric':
        if prev_combined_profile is None:
            return curr_distance_profile**curr_weight
        else:
            prev_combined_profile *= curr_distance_profile**curr_weight
            return prev_combined_profile
    elif combination_method == 'logarithmic':
        if prev_combined_profile is None:
            return curr_weight * np.log(curr_distance_profile)
        else:
            prev_combined_profile += curr_weight * np.log(curr_distance_profile)
            return prev_combined_profile
    else:
        raise ValueError(f"Invalid combination type {combination_method}")

