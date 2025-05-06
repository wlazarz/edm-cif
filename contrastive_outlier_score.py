import numpy as np
import pandas as pd
from typing import Callable, Union, List
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import (
    hamming, jaccard, cosine, euclidean, cityblock, braycurtis, chebyshev, minkowski
)


def eskin_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Eskin distance between two categorical vectors.
    """
    m = len(x)
    return sum([0 if xi == yi else 10 / (10 + freq) for xi, yi, freq in zip(x, y, [1]*m)])


def overlap_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute overlap similarity between two vectors.
    """
    return sum([1 if xi == yi else 0 for xi, yi in zip(x, y)]) / len(x)


def _prepare_distance_function(metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]) \
        -> Union[str, Callable[[np.ndarray, np.ndarray], float]]:
    """
    Select the appropriate distance function based on input.
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
    """
    if callable(dist_func):
        return np.array([[dist_func(ox, iy) for iy in inliers] for ox in outliers])
    return pairwise_distances(outliers, inliers, metric=dist_func)


def _aggregate_scores(distances: np.ndarray, inlier_labels: np.ndarray, strategy: str, n: int) -> List[float]:
    """
    Aggregate distance scores using the selected strategy.
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
    Compute average of n closest distances per inlier cluster.
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


def contrastive_outlier_score(
    df: Union[pd.DataFrame, np.ndarray],
    labels: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'cosine',
    strategy: str = 'average',
    n: int = 5
) -> float:
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


def one_hot_encoding(df):
    ohe_data = df.copy()
    for c in ohe_data.columns:
        col_mode = df[c].mode().values[0]
        df[c].fillna(col_mode, inplace=True)
        color_encoded = pd.get_dummies(ohe_data[c], prefix=c)
        ohe_data = pd.concat([ohe_data, color_encoded], axis=1).drop(columns=[c])

    return ohe_data