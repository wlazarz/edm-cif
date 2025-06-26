import numpy as np
import pandas as pd
from evaluation.object_distances import eskin_distance, overlap_distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, cdist
from collections import Counter
from sklearn.metrics import pairwise_distances
from typing import Optional, Dict, Tuple, Callable, Union, List


def calculate_mode(data: np.ndarray) -> np.ndarray:
    """
    Calculate the mode (most frequent value) for each column in a 2D array.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each column represents a feature and each row is a data point.

    Returns:
    -------
    np.ndarray
        A 1D numpy array containing the mode for each column.
    """
    modes = []
    for column in range(len(data[0])):
        unique, counts = np.unique(data[:, column], return_counts=True)
        index = np.argmax(counts)
        modes.append(unique[index])
    return np.array(modes)


def dunn_index(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the Dunn index, which is used to evaluate the quality of clustering.

    The Dunn index measures the ratio between the minimum inter-cluster distance
    and the maximum intra-cluster distance. A higher Dunn index indicates better clustering.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    Optional[float]
        The Dunn index, rounded to five decimal places, or None if no valid inter-cluster distance is found.
    """
    unique_labels = np.unique(labels)

    clusters = [X[labels == label] for label in unique_labels]
    intra_dists = [np.max(pdist(cluster)) if len(cluster) > 1 else 0 for cluster in clusters]
    inter_dists = [np.min(cdist(clusters[i], clusters[j])) for i in range(len(clusters)) for j in
                   range(i + 1, len(clusters))]

    if not inter_dists:
        return None

    return float(round(np.min(inter_dists) / np.max(intra_dists), 5))


def dunn_index_hamming(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the Dunn index with Hamming distance, useful for categorical data.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    Optional[float]
        The Dunn index, using Hamming distance, rounded to five decimal places, or None if no valid inter-cluster distance is found.
    """
    unique_labels = np.unique(labels)
    clusters = [X[labels == label] for label in unique_labels]

    intra_dists = [np.max(pdist(cluster, metric='hamming')) if len(cluster) > 1 else 0 for cluster in clusters]
    inter_dists = [np.min(cdist(clusters[i], clusters[j], metric='hamming')) for i in range(len(clusters)) for j in
                   range(i + 1, len(clusters))]

    if not inter_dists:
        return None

    return float(round(np.min(inter_dists) / np.max(intra_dists), 5))


def silhouette_score_calc(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the silhouette score for a clustering, a measure of how similar each point is to its own cluster
    compared to other clusters.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    conv_matrix : Optional[np.ndarray], optional (default=None)
        A precomputed conversion matrix (not used in this function).

    Returns:
    -------
    Optional[float]
        The silhouette score, rounded to five decimal places, or None if there is only one cluster.
    """
    return float(round(silhouette_score(X, labels), 5)) if len(np.unique(labels)) > 1 else None


def silhouette_score_hamming(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the silhouette score using Hamming distance for categorical data.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    Optional[float]
        The silhouette score, using Hamming distance, rounded to five decimal places, or None if there is only one cluster.
    """
    return float(round(silhouette_score(X, labels, metric='hamming'), 5)) if len(np.unique(labels)) > 1 else None


def calinski_harabasz_score_calc(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the Calinski-Harabasz index, which measures the ratio of the sum of between-cluster dispersion
    to within-cluster dispersion. Higher values indicate better clustering.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    Optional[float]
        The Calinski-Harabasz score, rounded to five decimal places, or None if there is only one cluster.
    """
    return float(round(calinski_harabasz_score(X, labels), 5)) if len(np.unique(labels)) > 1 else None


def entropy_(value_counts, base=2):
    """Computes the Shannon entropy for a set of value counts."""
    probabilities = value_counts / np.sum(value_counts)
    return -np.sum(probabilities * np.log(probabilities) / np.log(base))


def cluster_entropy(X, labels):
    """
    Computes the entropy-based evaluation of clustering quality for categorical data.

    :param X: NumPy array containing categorical variables (shape: [n_samples, n_features])
    :param labels: List or array containing cluster labels for each instance
    :return: Average cluster entropy (lower is better)
    """
    clusters = np.unique(labels)
    total_entropy = 0
    total_samples = len(X)

    for c in clusters:
        cluster_data = X[labels == c]
        cluster_size = len(cluster_data)

        feature_entropies = []

        for col in range(X.shape[1]):
            value_counts = np.unique(cluster_data[:, col], return_counts=True)[1]
            feature_entropy = entropy_(value_counts, base=2)
            feature_entropies.append(feature_entropy)

        cluster_entropy = np.mean(feature_entropies)

        weighted_entropy = (cluster_size / total_samples) * cluster_entropy
        total_entropy += weighted_entropy

    return float(round(total_entropy, 5))


def cluster_inconsistency(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the average inconsistency within clusters using the Hamming distance.

    Inconsistency is defined as the average distance between points in the same cluster.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    float
        The average inconsistency score, rounded to five decimal places.
    """
    unique_labels = np.unique(labels)
    inconsistencies = [np.mean(pdist(X[labels == label], metric='hamming')) if len(X[labels == label]) > 1 else 0
                       for label in unique_labels]
    return float(round(np.mean(inconsistencies), 5))


def cluster_separation(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the average separation between clusters using the Hamming distance.

    Separation is defined as the average distance between the centroids of different clusters.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    float
        The average separation score, rounded to five decimal places.
    """
    unique_labels = np.unique(labels)
    centroids = [np.mean(X[labels == label], axis=0) for label in unique_labels]
    return float(round(np.mean(pdist(np.array(centroids), metric='hamming')), 5))


def calinski_harabasz_hamming(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the Calinski-Harabasz index using Hamming distance for categorical data.

    The Calinski-Harabasz index measures the ratio of between-cluster dispersion to within-cluster dispersion.
    A higher value indicates better-defined clusters.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    Optional[float]
        The Calinski-Harabasz score using Hamming distance, rounded to five decimal places,
        or `np.nan` if the number of clusters is less than 2.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n = len(X)

    if k < 2:
        return np.nan

    overall_centroid = np.array([Counter(X[:, i]).most_common(1)[0][0] for i in range(X.shape[1])])
    centroids = {label: np.array([Counter(X[labels == label][:, i]).most_common(1)[0][0] for i in range(X.shape[1])])
                 for label in unique_labels}

    S_B = sum(len(X[labels == label]) * np.sum(centroids[label] != overall_centroid) for label in unique_labels)
    S_W = sum(np.sum(X[labels == label] != centroids[label]) for label in unique_labels)

    return float(round((S_B / S_W) * ((n - k) / (k - 1)), 5))


def davies_bouldin_score_calc(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute the Davies-Bouldin score, a measure of the average similarity ratio of each cluster.

    The Davies-Bouldin index is used to evaluate clustering quality; lower values indicate better clustering.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    Optional[float]
        The Davies-Bouldin score, rounded to five decimal places, or `None` if there is only one cluster.
    """
    return float(round(davies_bouldin_score(X, labels), 5)) if len(np.unique(labels)) > 1 else None


def davies_bouldin_hamming(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Davies-Bouldin score using Hamming distance for categorical data.

    The Davies-Bouldin score measures the average similarity ratio of each cluster,
    with lower values indicating better clustering.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points.

    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    Returns:
    -------
    float
        The Davies-Bouldin score using Hamming distance, rounded to five decimal places.
    """
    unique_labels = np.unique(labels)
    clusters = [X[labels == label] for label in unique_labels]
    centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    intra_dists = [np.mean(pdist(cluster, metric='hamming')) for cluster in clusters]
    inter_dists = cdist(np.array(centroids), np.array(centroids), metric='hamming')
    np.fill_diagonal(inter_dists, np.inf)
    db_scores = [(intra_dists[i] + intra_dists[j]) / inter_dists[i, j] for i in range(len(clusters)) for j in
                 range(len(clusters)) if i != j]
    return float(round(np.mean(db_scores), 5))


def profit(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the profit-based evaluation for clustering quality.

    The profit-based evaluation measures how well the data points in each cluster are distributed.
    A higher profit score indicates better clustering quality.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points with shape [n_samples, n_features].

    labels : np.ndarray
        A 1D numpy array or list containing cluster labels for each data point.

    Returns:
    -------
    float
        The profit evaluation score, rounded to five decimal places. Higher values are better.
        Returns 0 if the total power is 0.
    """
    columns_count = X.shape[1]
    clusters = np.unique(labels)
    total_power = 0
    profit = 0

    for c in clusters:
        cluster_data = X[labels == c]
        S_c = len(cluster_data) * columns_count
        W_c = len(np.unique(cluster_data))
        power_c = S_c * columns_count
        P_c = S_c * power_c / W_c if W_c != 0 else 0
        profit += P_c
        total_power += power_c

    return float(round(profit / total_power, 5)) if total_power != 0 else 0


def M1(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes a clustering evaluation metric (M1).

    M1 measures the consistency of the clustering by considering the distribution
    of unique values in the data and their occurrence within each cluster.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points with shape [n_samples, n_features].

    labels : np.ndarray
        A 1D numpy array or list containing cluster labels for each data point.

    Returns:
    -------
    float
        The M1 evaluation score, rounded to five decimal places. Higher values indicate better clustering.
        Returns 0 if there are no clusters.
    """
    clusters = np.unique(labels)
    k = len(clusters)
    result = 0

    for c in clusters:
        cluster_mask = labels == c
        cluster_data = X[cluster_mask]
        cluster_size = len(cluster_data)

        X_flat = X.ravel()
        unique_values, counts = np.unique(X_flat, return_counts=True)
        count_dict = dict(zip(unique_values, counts))

        for t in cluster_data:
            sum_ = 0
            for i in t:
                sum_ += count_dict.get(i, 0)
            result = sum_ / cluster_size

    return float(round(result / k, 5)) if k > 0 else 0


def M2(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes a clustering evaluation metric (M2).

    M2 measures the dissimilarity between the distribution of values in the entire dataset
    and the distribution within each cluster. A lower value indicates better clustering quality.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points with shape [n_samples, n_features].

    labels : np.ndarray
        A 1D numpy array or list containing cluster labels for each data point.

    Returns:
    -------
    float
        The M2 evaluation score, rounded to five decimal places. Lower values are better.
        Returns 0 if there are no clusters.
    """
    clusters = np.unique(labels)
    k = len(clusters)
    result = 0

    unique_values, value_counts_in_D = np.unique(X, return_counts=True)
    value_counts_dict = dict(zip(unique_values, value_counts_in_D))

    for c in clusters:
        cluster_mask = labels == c
        cluster_data = X[cluster_mask]

        unique_vals_Ci, counts_Ci = np.unique(cluster_data, return_counts=True)
        P_v_log_sum = np.sum(
            (counts_Ci / np.array([value_counts_dict[v] for v in unique_vals_Ci])) *
            np.log(counts_Ci / np.array([value_counts_dict[v] for v in unique_vals_Ci]))
        )

        result += P_v_log_sum

    return float(round(-result / k, 5)) if k > 0 else 0


def M3(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Computes two metrics based on the frequency of unique values within each cluster:
    - Metric M3: Sum of occurrences above a cutoff point based on frequency differences.
    - Metric M4: Ratio of occurrences above and below the cutoff point.

    For each cluster, the unique values within the cluster are counted and sorted based on their occurrences.
    A cutoff point is determined by the largest difference in occurrences between consecutive values.
    The sum of occurrences above the cutoff point is then divided by the product of the number of unique values
    in the cluster and the cluster size. The result is averaged over all clusters.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array containing data points (shape: [n_samples, n_features]).

    labels : np.ndarray
        A 1D numpy array containing the cluster labels for each data point.

    Returns:
    -------
    Tuple[float, float]
        - M3: A metric calculated for each cluster based on the cutoff point.
        - M4: A metric calculated based on the ratio of occurrences above and below the cutoff point.
    """
    clusters = np.unique(labels)
    k = len(clusters)
    result_m3 = 0
    result_m4 = 0

    for c in clusters:
        cluster_mask = labels == c
        cluster_data = X[cluster_mask]

        unique_vals, counts = np.unique(cluster_data, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]

        diffs = np.diff(sorted_counts)
        cutoff_point = np.argmax(diffs)

        sum_above_cutoff = sorted_counts[:cutoff_point + 1].sum()
        sum_below_cutoff = sorted_counts[cutoff_point + 1:].sum()

        cluster_power = cluster_data.size
        unique_values_in_Ci = len(unique_vals)

        result_m3 += sum_above_cutoff / (unique_values_in_Ci * cluster_power)
        if sum_below_cutoff > 0:
            result_m4 += sum_above_cutoff / sum_below_cutoff

    return (float(round(result_m3 / k, 5)) if k > 0 else 0,
            float(round(result_m4 / k, 5)) if k > 0 else 0)


def hamming_ratio(X_values: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Hamming ratio, which is the ratio of intra-cluster Hamming distance to inter-cluster Hamming distance.

    For each cluster, the average pairwise Hamming distance within the cluster is computed.
    The cluster centers are then determined using the mode of the values within each cluster, and the average pairwise
    Hamming distance between the cluster centers is calculated. Finally, the ratio of the intra-cluster distance to
    the inter-cluster distance is returned.

    Parameters:
    ----------
    X_values : np.ndarray
        A 2D numpy array containing the data points (shape: [n_samples, n_features]).

    labels : np.ndarray
        A 1D numpy array containing the cluster labels for each data point.

    Returns:
    -------
    float
        The Hamming ratio, rounded to five decimal places. Returns 0 if the inter-cluster distance is 0.
    """
    clusters = np.unique(labels)

    A = 0
    k = len(clusters)
    cluster_centers = []

    for c in clusters:
        cluster_data = X_values[labels == c]
        n = len(cluster_data)

        if n > 1:
            pairwise_hamming = np.mean(pdist(cluster_data, metric='hamming'))
            A += pairwise_hamming

        cluster_centers.append(calculate_mode(cluster_data))

    A /= k
    cluster_centers = np.array(cluster_centers)
    B = np.mean(pdist(cluster_centers, metric='hamming')) if k > 0 else 0

    return float(round(A / B, 5)) if B > 0 else 0


def clustering_metrics(df: pd.DataFrame, df_ohe: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute a variety of clustering evaluation metrics for the given data and clustering labels.

    This function calculates various clustering metrics including silhouette score,
    Calinski-Harabasz score, Davies-Bouldin score, Dunn index, cluster entropy,
    cluster inconsistency, cluster separation, profit, Hamming ratio, and other custom metrics.

    Parameters:
    ----------
    df : pd.DataFrame or np.ndarray
        A DataFrame or numpy array containing the original data points.

    df_ohe : pd.DataFrame or np.ndarray
        A DataFrame or numpy array containing the one-hot encoded data points.

    labels : np.ndarray
        A 1D numpy array containing the cluster labels for each data point.

    Returns:
    -------
    Dict[str, float]
        A dictionary where each key is a metric name, and the corresponding value is the metric score.
    """
    labels = np.array(labels)
    X = df.values if isinstance(df, pd.DataFrame) else df
    X_ohe = df_ohe.values if isinstance(df_ohe, pd.DataFrame) else df_ohe

    if X.dtype == bool:
        X = X.astype(int)
    if X_ohe.dtype == bool:
        X_ohe = X_ohe.astype(int)

    m3_res, m4_res = M3(X, labels)

    return {
        "Silhouette Score": silhouette_score_calc(X_ohe, labels),
        "Silhouette Score (Hamming)": silhouette_score_hamming(X_ohe, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score_calc(X_ohe, labels),
        "Calinski-Harabasz Score (Hamming)": calinski_harabasz_hamming(X_ohe, labels),
        "Davies-Bouldin Score": davies_bouldin_score_calc(X_ohe, labels),
        "Davies-Bouldin Index (Hamming)": davies_bouldin_hamming(X_ohe, labels),
        "Dunn Index": dunn_index(X_ohe, labels),
        "Dunn Index Hamming": dunn_index_hamming(X, labels),
        "Cluster Entropy": cluster_entropy(X, labels),
        "Cluster Inconsistency": cluster_inconsistency(X_ohe, labels),
        "Cluster Separation": cluster_separation(X_ohe, labels),
        "Profit": profit(X, labels),
        "Hamming ratio": hamming_ratio(X_ohe, labels),
        "M1": M1(X, labels),
        "M2": M2(X, labels),
        "M3": m3_res,
        "M4": m4_res
    }


################################################ WITH CUSTOM MEASURE#################################################

def dunn_user_metric(labels: np.ndarray, con_matrix: np.ndarray) -> Optional[float]:
    """
    Calculates the Dunn Index for a categorical dataset based on a distance matrix.

    The Dunn Index is a measure of clustering quality, defined as the ratio of the minimum
    inter-cluster distance to the maximum intra-cluster distance. A higher Dunn Index indicates better clustering.

    Parameters:
    ----------
    labels : np.ndarray
        A 1D numpy array of cluster labels for each data point.

    con_matrix : np.ndarray
        A 2D numpy array representing the distance matrix between data points.

    Returns:
    -------
    Optional[float]
        The Dunn Index, rounded to five decimal places, or None if no valid inter-cluster distances exist.
    """
    unique_labels = np.unique(labels)

    intra_dists = []
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    for label, indices in cluster_indices.items():
        if len(indices) > 1:
            intra_dists.append(np.max(con_matrix[np.ix_(indices, indices)]))
        else:
            intra_dists.append(0.0)

    inter_dists = []
    cluster_labels = list(cluster_indices.keys())

    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            indices_i = cluster_indices[cluster_labels[i]]
            indices_j = cluster_indices[cluster_labels[j]]
            dist = np.min(con_matrix[np.ix_(indices_i, indices_j)])
            if dist > 0:
                inter_dists.append(dist)

    if not inter_dists or np.max(intra_dists) == 0:
        return None

    dunn_index = np.min(inter_dists) / np.max(intra_dists)
    return float(round(dunn_index, 5))


def calinski_harabasz_user_metric(X: np.ndarray, labels: np.ndarray, con_matrix: np.ndarray) -> Optional[float]:
    """
    Compute the Calinski-Harabasz Index based on a distance matrix.

    The Calinski-Harabasz index is used to evaluate clustering quality, based on the ratio of
    between-cluster dispersion to within-cluster dispersion. A higher value indicates better clustering.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array of data points (shape: [n_samples, n_features]).

    labels : np.ndarray
        A 1D numpy array containing cluster labels for each data point.

    con_matrix : np.ndarray
        A 2D numpy array representing the distance matrix between data points.

    Returns:
    -------
    Optional[float]
        The Calinski-Harabasz Index, rounded to five decimal places, or None if there are fewer than 2 clusters.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(X)

    if n_clusters < 2:
        return None

    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    cluster_medoids = {}

    for label, indices in cluster_indices.items():
        if len(indices) > 1:
            intra_dists = con_matrix[np.ix_(indices, indices)].sum(axis=1)
            medoid_index = indices[np.argmin(intra_dists)]
        else:
            medoid_index = indices[0]
        cluster_medoids[label] = medoid_index

    S_W = sum(np.sum(con_matrix[np.ix_(indices, [medoid])]) for label, indices in cluster_indices.items()
              for medoid in [cluster_medoids[label]])

    overall_medoid = np.argmin(np.sum(con_matrix, axis=1))
    S_B = sum(
        len(cluster_indices[label]) * con_matrix[cluster_medoids[label], overall_medoid] for label in unique_labels)

    numerator = (S_B / (n_clusters - 1))
    denominator = (S_W / (n_samples - n_clusters))
    CH_score = numerator / denominator

    return float(round(CH_score, 5))


def davies_bouldin_user_metric(labels: np.ndarray, con_matrix: np.ndarray) -> Optional[float]:
    """
    Calculates the Davies-Bouldin Index for a categorical dataset based on a distance matrix.

    The Davies-Bouldin Index is a measure of clustering quality, defined as the average similarity
    between each cluster and its most similar one. A lower Davies-Bouldin Index indicates better clustering.

    Parameters:
    ----------
    labels : np.ndarray
        A 1D numpy array containing the cluster labels for each data point.

    con_matrix : np.ndarray
        A 2D numpy array representing the distance matrix between data points.

    Returns:
    -------
    Optional[float]
        The Davies-Bouldin Index, rounded to five decimal places, or `None` if there are fewer than 2 clusters.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return None

    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    cluster_medoids = {}

    for label, indices in cluster_indices.items():
        if len(indices) > 1:
            intra_dists = con_matrix[np.ix_(indices, indices)].sum(axis=1)
            medoid_index = indices[np.argmin(intra_dists)]
        else:
            medoid_index = indices[0]
        cluster_medoids[label] = medoid_index

    S = {}
    for label, indices in cluster_indices.items():
        medoid = cluster_medoids[label]
        S[label] = np.mean(con_matrix[np.ix_(indices, [medoid])]) if len(indices) > 1 else 0

    R = np.zeros((n_clusters, n_clusters))
    epsilon = 1e-10

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i != j:
                medoid_i = cluster_medoids[label_i]
                medoid_j = cluster_medoids[label_j]
                M_ij = con_matrix[medoid_i, medoid_j]
                if M_ij > epsilon:
                    R[i, j] = (S[label_i] + S[label_j]) / M_ij
                else:
                    R[i, j] = np.inf

    R = np.where(np.isinf(R), np.nan, R)
    D_B = np.nanmean(np.nanmax(R, axis=1))

    return float(round(D_B, 5))


def silhouette_score_user_metric(labels: np.ndarray, con_matrix: Optional[np.ndarray] = None) -> float:
    """
    Calculates the Silhouette Score for a categorical dataset based on a distance matrix.

    The Silhouette Score is a measure of how similar each data point is to its own cluster compared
    to other clusters. A higher score indicates better clustering.

    Parameters:
    ----------
    labels : np.ndarray
        A 1D numpy array containing the cluster labels for each data point.

    con_matrix : np.ndarray, optional
        A 2D numpy array representing the distance matrix between data points. If not provided,
        it assumes that the distances are already provided in the matrix.

    Returns:
    -------
    float
        The Silhouette Score, rounded to five decimal places.
    """
    unique_labels = np.unique(labels)
    n_samples = len(labels)

    a = np.zeros(n_samples)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    for label, indices in cluster_indices.items():
        for i in indices:
            if len(indices) > 1:
                a[i] = np.mean(con_matrix[i, indices[indices != i]])
            else:
                a[i] = 0.0

    b = np.zeros(n_samples)
    for i in range(n_samples):
        other_clusters = [label for label in unique_labels if label != labels[i]]
        b[i] = np.min([np.mean(con_matrix[i, cluster_indices[label]]) for label in other_clusters])

    s = (b - a) / np.maximum(a, b)
    s = np.nan_to_num(s)

    return float(round(np.mean(s), 5))

################################################ OUTLIERS EVALUATION #################################################

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
            return overlap_distance
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


if __name__ == '__main__':
    pass