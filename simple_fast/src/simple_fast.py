import numpy as np
from .utils import partition_find_min_x, mass_pre, mass, euc_dist, cosine_dist, EPS
from typing import Literal


def simple_fast2(
    data:np.ndarray, 
    query:np.ndarray, 
    window_size, 
    distance_metric: Literal['cosine', 'euclidean'] = 'euclidean', 
    k = 1):
    """
    For every sliding window in *data*, find its *k* nearest windows in *query* 
    
    It must be true that `query.shape[1] == data.shape[1]` or `query.shape[0] == data.shape[0]`
    - If not `query.shape[1] == data.shape[1]`, then *data* and *query* will be transposed

    Args:
        data (np.ndarray):
        query (np.ndarray):
        window_size (_type_): The number of time points (rows) for one window
        distance_metric (Literal[&#39;cosine&#39;, &#39;euclidean&#39;], optional): The metric to use to compare individual windows. Defaults to 'euclidean'.
        k (int, optional): The number of_. Defaults to 1.

    Raises:
        ValueError: If not `query.shape[1] == data.shape[1]` or `query.shape[0] == data.shape[0]`
        ValueError: If the *window_size* is larger than the time dimension of either *data* or *query*

    Returns:
        tuple[ndarray]: An ndarray of shortest distances for each sliding data window and the corresponding indices in query
        - If k = 1, then a 1d array is returned with shape (# of data windows, ) 
        - If k > 1, then a 2d array is returned with shape *(k, # of data windows)*
        - The index matrix is of dtype *uint64*
    """       
    """For each window in *data*, find the closest *k* windows in the query
    
    """
    if query.shape[1] != data.shape[1]:
        data = data.T
        query = query.T
    if query.shape[1] != data.shape[1]:
        raise ValueError( f"incompatible dimensions: data {data.shape}, query {query.shape}, " )
    n, dim = query.shape
    if window_size > n or window_size > data.shape[0]:
        raise ValueError(f"The window size {window_size} to large for the input time dimensions: data {data.shape}, query {query.shape}")

    dist_func = cosine_dist if distance_metric == 'cosine' else euc_dist
    matrix_profile_length = data.shape[0] - window_size + 1
    
    k = min(k, query.shape[0] - window_size + 1)
    matrix_profile = np.empty(matrix_profile_length, dtype = data.dtype) if k == 1 else np.empty((k , matrix_profile_length), dtype = data.dtype)
    profile_index = np.zeros(matrix_profile_length, dtype = np.uint64) if k == 1 else np.empty((k , matrix_profile_length), dtype = np.uint64)

    # compute the first dot-product for the data and query
    X, sumx2 = mass_pre(data, window_size)
    _, z0, _ = mass(X, query[:window_size], data.shape[0], window_size, dim, sumx2, dist_func)
    # print(f"X0 {X.shape}")
    # print(f"sumx2_0 {sumx2.shape}")
    # print(f"z0 {z0.shape}" )
    
    # compute the first distance profile
    X, sumx2 = mass_pre(query, window_size)
    distance_profile, z, sumy2 = mass(X, data[:window_size], n, window_size, dim, sumx2, dist_func)
    dropval = data[0]
    # print(f"\nX1 {X.shape}")
    # print(f"sumx2_1 {sumx2.shape}, sumy2 {sumy2.shape}, prof {distance_profile.shape}")
    # print(f"z {z.shape}" )
    # print(f"drop_val {dropval.shape}" )
    
    if k == 1:
        idx = np.argmin(distance_profile)
        profile_index[0] = idx
        matrix_profile[0] = distance_profile[idx] 
    else:
        min_dists, indices = partition_find_min_x(distance_profile, k)
        matrix_profile[:, 0] = min_dists
        profile_index[:, 0] = indices

    # compute the rest of the matrix profile
    nz, _ = z.shape
    for i in range(1, matrix_profile_length):
        subsequence = data[i : i + window_size]
        sumy2 = sumy2 - dropval ** 2 + subsequence[-1] ** 2
        z[1:nz] = (
            z[: nz - 1]
            + subsequence[-1] * query[window_size : window_size + nz - 1]
            - dropval * query[: nz - 1]
        )  
        z[0] = z0[i]
        dropval = subsequence[0]

        # Compute distance profile
        distance_profile =  dist_func(sumx2, sumy2, z)
        if k == 1:
            idx = np.argmin(distance_profile)
            profile_index[i] = idx
            matrix_profile[i] = distance_profile[idx]
        else:
            min_dists, indices = partition_find_min_x(distance_profile, k)
            profile_index[:, i] = indices
            matrix_profile[:, i] = min_dists
            
    return matrix_profile, profile_index


