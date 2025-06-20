import numpy as np
from .utils import *
from .simple_fast import simple_fast2
from typing import Sequence, Literal

def multi_fast(
    data_features: Sequence[np.ndarray], 
    query_features: Sequence[np.ndarray], 
    window_size: int, 
    weights: np.ndarray, 
    distance_metric: Literal['cosine', 'euclidean'] = 'cosine', 
    combination_method: Literal['weighted', 'geometric', 'harmonic', 'rms', 'softmax'] = 'weighted', 
    k: int = 1):  
    """For every sliding window in *data*, find its *k* nearest windows in *query* using multiple data and query features
    
    Note, it is initially assumed that the row dimension is the time dimension. If the first data feature and query feature have different column dimensions, then *all* features will be transposed.
    
    Args:
        data_features (Sequence[ndarray]): A sequence of data features that share a time-dimension.
        query_features (Sequence[ndarray]): A sequence of query features that share a time-dimension.
        window_size (int): The number of time points (rows) for one window
        weights (ndarray): An array of weights associated with each feature. They will normalized in-place if they do not sum to 1.
        distance_metric ((Literal[&#39;cosine&#39;, &#39;euclidean&#39;], optional): The metric to use to compare individual windows. Defaults to 'cosine'.
        combination_method (Literal[&#39;weighted&#39;, &#39;geometric&#39;, &#39;harmonic&#39;, &#39;rms&#39;, &#39;softmax&#39;], optional): A method for combining the distance values of each feature. Defaults to 'weighted'.
        k (int, optional): The number of nearest windows to find in the query. Defaults to 1.

    Raises:
        ValueError: If *data_features* or *query_features* is an empty sequence
        ValueError: If not `len(data features) == len(query_features) == len(weights)`
        ValueError: If not all *data_features* share the same time-dimension (number of rows)
        ValueError: If not all *query_features* share the same time-dimension (number of rows)
        ValueError: If any (`data_feature[i], query_features[i]`) pair do not share the same feature-dimension (number of columns). 
            Note, different data-query pairs may have different feature dimensions.
        ValueError: If `sum(weights) == 0`weights is equal to 0
        ValueError: If the *window_size* is larger than the time dimension of either *data* or *query*
    
    Returns:
        tuple[ndarray]: An ndarray of shortest distances for each sliding window in *data* and the corresponding indices in *query*
        - If k = 1, then a 1d array is returned with shape (# of data windows, ) 
        - If k > 1, then a 2d array is returned with shape *(k, # of data windows)*
        - The corresponding indices are of dtype *uint64*
        
    Example:
        Say there are `n` features and we are finding the distance between the first data window and the first query window.
        The combination method is 'weighted' the distance metric is 'cosine'. The window distance is equal to:
        - `weights[0] * cosine_dist(data_feature[0][:window], query_feature[0][:window]) +`
        
            `weights[1] * cosine_dist(data_feature[1][:window], query_feature[1][:window]) + ...`
        
            weights[n] * cosine_dist(data_feature[n][:window], query_feature[n][:window])
    """    

    nqueries = len(query_features)
    if nqueries == 0:
        raise ValueError("Got an empty list for the queries")
    if nqueries == 1:
        return simple_fast2(data_features[0], query_features[0], window_size=window_size, distance_metric = distance_metric, k = k)
    if not nqueries == len(data_features) or not nqueries == len(weights):
        raise ValueError(
            f"Must have the same number of query features, data features and weights, got {nqueries}, {len(data_features)}, {len(weights)}"
        )
    
    # Normalize weights 
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        raise ValueError("The sum of the weights is 0")
    if weight_sum != 1:
        weights /= weight_sum
    
    # check dimension compatibilies 
    first_query = query_features[0]
    first_data = data_features[0]
    ntimes_data = first_data.shape[0]
    ntimes_query = first_query.shape[0]
    transpose = False
    if first_query.shape[1] != first_data.shape[1]:
        if ntimes_data != ntimes_query:
            print(f"times_d {ntimes_data}, times_q {ntimes_query}")
            raise ValueError(f"Data and query matrices at index 0 have incompatible dimensions: data {first_data.shape}, query {first_query.shape}")
        transpose = True
        ntimes_data = first_data.shape[1]
        ntimes_query = first_query.shape[1]
    if window_size > ntimes_data or window_size > ntimes_query:
        raise ValueError(f"The window size {window_size} to large for the input time dimensions: data {ntimes_data}, query {ntimes_query}")
    
    dist_func = cosine_dist if distance_metric == 'cosine' else euc_dist
    accumulator_func, post_weight_func = get_score_combinator(combination_method)
    nz = ntimes_query - window_size + 1 # number of windows in query
    
    matrix_profile_length =  ntimes_data - window_size + 1
    k = min(nz, k)
    matrix_profile = np.empty(matrix_profile_length, dtype = first_data.dtype) if k == 1 else np.empty((k , matrix_profile_length), dtype = first_data.dtype)
    profile_index = np.zeros(matrix_profile_length, dtype = np.uint64) if k == 1 else np.empty((k , matrix_profile_length), dtype = np.uint64)
    
    sumx2s = []
    z0s = []
    zs = []
    sumy2s = []
    dropvals = []
    weighted_distance_profile = np.zeros(nz, dtype = first_data.dtype)
    for i in range(nqueries):
        curr_data = data_features[i]
        curr_query = query_features[i]
        if transpose:
            curr_data = curr_data.T
            curr_query = curr_query.T
        dim = curr_data.shape[1]
        if curr_data.shape[0] != ntimes_data:
            raise ValueError(
                f"The number of time dimension of the first data matrix and the data matrix {i} are not equal ({ntimes_data} vs. {curr_data.shape[0]})"
            )
        if curr_query.shape[0] != ntimes_query:
            raise ValueError(
                f"The number of time dimension of the first query  and the query {i} are not equal ({ntimes_query} vs. {curr_query.shape[0]})"
            )
        if curr_query.shape[1] != dim:
            raise ValueError(
                f"Data and query matrices at index {i} have incompatible dimensions: data {curr_data.shape}, query {curr_query.shape}"
            )

        # compute the distance profile for first data window
        X, sumx2 = mass_pre(curr_data, window_size)
        _, z0, _ = mass(X, curr_query[:window_size], curr_data.shape[0], window_size, dim, sumx2, dist_func)
        
        X, sumx2 = mass_pre(curr_query, window_size)
        distance_profile, z, sumy2 = mass(X, curr_data[:window_size], ntimes_query, window_size, dim, sumx2, dist_func)
        # if distance_metric== 'euclidean':
        #         distance_profile = z_score_norm(distance_profile)
        dropval = curr_data[0]
        accumulator_func(weighted_distance_profile, distance_profile, weights[i])
        
        sumx2s.append(sumx2)
        z0s.append(z0)
        zs.append(z)
        sumy2s.append(sumy2)
        dropvals.append(dropval)
    
    post_weight_func(weighted_distance_profile)
    
    if k == 1:
        idx = np.argmin(weighted_distance_profile)
        profile_index[0] = idx
        matrix_profile[0] = weighted_distance_profile[idx]
    else:
        min_dists, idx = partition_find_min_x(weighted_distance_profile, k)
        profile_index[:, 0] = idx
        matrix_profile[:, 0] = min_dists
        
    for i in range(1, matrix_profile_length):
        weighted_distance_profile = np.zeros(nz, dtype = first_data.dtype)
        for j in range(nqueries):
            data = data_features[j]
            query = query_features[j]
            sumx2 = sumx2s[j]
            sumy2 = sumy2s[j]
            z0 = z0s[j]
            z = zs[j]
            dropval = dropvals[j]
            
            subsequence = data[i : i + window_size]
            sumy2 = sumy2 - (dropval ** 2) + (subsequence[-1] ** 2)
            z[1:nz] = (
                z[: nz - 1]
                + subsequence[-1] * query[window_size : window_size + nz - 1]
                - dropval * query[: nz - 1]
            )
            z[0] = z0[i]
            dropvals[j] = subsequence[0]
            sumy2s[j] = sumy2
            distance_profile = dist_func(sumx2, sumy2, z)
            accumulator_func(weighted_distance_profile, distance_profile, weights[j])
        
        post_weight_func(weighted_distance_profile)
 
        if k == 1:
            idx = np.argmin(weighted_distance_profile)
            profile_index[i] = idx
            matrix_profile[i] = weighted_distance_profile[idx] 
        else:
            min_dists, indices = partition_find_min_x(weighted_distance_profile, k)
            profile_index[:, i] = indices
            matrix_profile[:, i] = min_dists

    return matrix_profile, profile_index
