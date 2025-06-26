import numpy as np
import pandas as pd
from typing import Callable, Union, List
from sklearn.metrics import pairwise_distances


def eskin_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Eskin distance between two categorical vectors.

    The Eskin distance is a measure of dissimilarity between two categorical vectors,
    with higher penalties for mismatches in attributes with many unique values.
    It is calculated as the sum of 1 / (10 + frequency) for each differing feature,
    where frequency is set to 1 for all features here.

    Parameters:
    ----------
    x : np.ndarray
        A 1D numpy array representing the first categorical vector.

    y : np.ndarray
        A 1D numpy array representing the second categorical vector.

    Returns:
    -------
    float
        The Eskin distance between the two vectors.
    """
    m = len(x)
    return sum([0 if xi == yi else 10 / (10 + freq) for xi, yi, freq in zip(x, y, [1] * m)])


def overlap_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the overlap similarity between two vectors.

    The overlap similarity is the ratio of matching elements in the two vectors to the total number of elements.

    Parameters:
    ----------
    x : np.ndarray
        A 1D numpy array representing the first vector.

    y : np.ndarray
        A 1D numpy array representing the second vector.

    Returns:
    -------
    float
        The overlap similarity between the two vectors (range [0, 1]).
    """
    return sum([1 if xi == yi else 0 for xi, yi in zip(x, y)]) / len(x)


def _prepare_distance_function(metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]) \
        -> Union[str, Callable[[np.ndarray, np.ndarray], float]]:
    """
    Select the appropriate distance function based on the input.

    If the `metric` is a string, the function selects a built-in distance function or an Eskin-based metric.
    If the `metric` is already a callable function, it is returned as is.

    Parameters:
    ----------
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        The metric used to compute distances. It can either be a string (e.g., 'overlap', 'eskin', or others)
        or a user-provided callable function.

    Returns:
    -------
    Union[str, Callable[[np.ndarray, np.ndarray], float]]
        The appropriate distance function corresponding to the input metric.

    Raises:
    ------
    ValueError
        If the provided metric is not recognized or supported.
    """
    if isinstance(metric, str):
        metric = metric.lower()
        if metric == 'overlap':
            return lambda x, y: 1 - overlap_similarity(x, y)
        elif metric == 'eskin':
            return eskin_distance
        elif metric in {'hamming', 'jaccard', 'cosine', 'euclidean', 'manhattan', 'cityblock', 'chebyshev', 'minkowski',
                        'braycurtis'}:
            return metric
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return metric


def _compute_distances(outliers: np.ndarray, inliers: np.ndarray,
                       dist_func: Union[str, Callable[[np.ndarray, np.ndarray], float]]) -> np.ndarray:
    """
    Compute pairwise distances between outliers and inliers.

    This function computes distances between each pair of outlier and inlier using the provided distance function.

    Parameters:
    ----------
    outliers : np.ndarray
        A 2D numpy array where each row is an outlier.

    inliers : np.ndarray
        A 2D numpy array where each row is an inlier.

    dist_func : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        The distance function or string that defines the metric to compute the pairwise distances. It can be a built-in
        string metric (e.g., 'euclidean', 'manhattan', etc.) or a custom distance function.

    Returns:
    -------
    np.ndarray
        A 2D numpy array of pairwise distances between each outlier and inlier.
    """
    if callable(dist_func):
        return np.array([[dist_func(ox, iy) for iy in inliers] for ox in outliers])
    return pairwise_distances(outliers, inliers, metric=dist_func)


def _aggregate_scores(distances: np.ndarray, inlier_labels: np.ndarray, strategy: str, n: int) -> List[float]:
    """
    Aggregate distance scores using the selected strategy.

    This function computes aggregated distance scores for each data point in the outlier set based on the specified
    strategy. The strategies supported are:
    - 'average': Compute the average distance.
    - 'median': Compute the median distance.
    - 'n_closest': Compute the average of the `n` closest distances within each inlier cluster.

    Parameters:
    ----------
    distances : np.ndarray
        A 2D numpy array where each row represents the distances between an outlier and all inliers.

    inlier_labels : np.ndarray
        A 1D numpy array containing the labels of the inliers, used to group distances by inlier cluster.

    strategy : str
        The aggregation strategy to use. Options are 'average', 'median', or 'n_closest'.

    n : int
        The number of closest distances to consider for the 'n_closest' strategy.

    Returns:
    -------
    List[float]
        A list of aggregated scores for each outlier, one score per outlier.

    Raises:
    ------
    ValueError
        If the provided strategy is not recognized.
    """
    scores = []
    for dist_row in distances:
        if strategy == 'average':
            score = np.mean(dist_row)
        elif strategy == 'median':
            score = np.median(dist_row)
        elif strategy == 'n_closest':
            score = _n_closest_score(dist_row, inlier_labels, n)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        scores.append(score)
    return scores


def _n_closest_score(dist_row: np.ndarray, inlier_labels: np.ndarray, n: int) -> float:
    """
    Compute the average of the `n` closest distances per inlier cluster.

    This function computes the average of the `n` smallest distances within each inlier cluster, where `n` is
    the number of closest distances to consider. The distances are grouped by inlier cluster, and the `n` smallest
    distances within each cluster are selected for averaging.

    Parameters:
    ----------
    dist_row : np.ndarray
        A 1D numpy array representing the distances between a specific outlier and all inliers.

    inlier_labels : np.ndarray
        A 1D numpy array containing the labels of the inliers, used to group distances by inlier cluster.

    n : int
        The number of closest distances to consider for each inlier cluster.

    Returns:
    -------
    float
        The average of the `n` closest distances across all inlier clusters.
    """
    cluster_scores = []
    for label in np.unique(inlier_labels):
        cluster_distances = dist_row[inlier_labels == label]
        if len(cluster_distances) == 0:
            continue
        k = min(n, len(cluster_distances))
        closest = np.partition(cluster_distances, k - 1)[:k]
        cluster_scores.extend(closest)

    return np.mean(cluster_scores) if cluster_scores else np.nan


def contrastive_outlier_score(df: Union[pd.DataFrame, np.ndarray], labels: np.ndarray,
                              metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'cosine',
                              strategy: str = 'average', n: int = 5) -> float:
    """
    Calculate the Contrastive Outlier Score (COS) for evaluating how distinct outliers are from inliers.

    Parameters
    ----------
    df : pd.DataFrame or np.ndarray
        Input dataset containing features.
    labels : np.ndarray
        Array indicating outliers (-1) and inliers (other values).
    metric : str or callable
        Distance or similarity function to compute pairwise distances.
    strategy : str
        Aggregation method: 'average', 'median', or 'n_closest'.
    n : int
        Number of closest neighbors to use for 'n_closest' strategy.

    Returns
    -------
    float
        Contrastive outlier score (higher = more distinct outliers).
    """
    if isinstance(df, pd.DataFrame):
        df = df.values

    labels = np.array(labels)
    outliers = df[labels == -1]
    inliers = df[labels != -1]
    inlier_labels = labels[labels != -1]

    dist_func = _prepare_distance_function(metric)
    distances = _compute_distances(outliers, inliers, dist_func)
    scores = _aggregate_scores(distances, inlier_labels, strategy, n)

    return round(float(np.nanmean(scores)), 4)

