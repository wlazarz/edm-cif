from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Union, Optional
from scipy.spatial.distance import hamming

NpList = Union[np.ndarray, List]


class Clustering(ABC):
    """
    Abstract base class for clustering algorithms.

    This class defines the basic structure for clustering algorithms and methods to detect outliers.
    Specific clustering algorithms should subclass this class and implement the `fit_predict` method.

    Attributes:
    ----------
    labels : Optional[NpList]
        The cluster labels for each sample (or None before fitting the model).

    outliers : Optional[NpList]
        The outliers identified in the dataset (or None if not identified).

    model : Optional[object]
        The underlying clustering model object (or None before fitting).
    """

    def __init__(self):
        """
        Initializes the Clustering class, setting up initial attributes for labels, outliers, and the model.
        """
        self.labels: Optional[NpList] = None
        self.outliers: Optional[NpList] = None
        self.model: Optional[object] = None

    @abstractmethod
    def fit_predict(self, X: pd.DataFrame) -> List:
        """
        Abstract method to be implemented by subclasses for clustering the data.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data to cluster.

        Returns:
        -------
        List
            The cluster labels for each sample.
        """
        pass

    def cluster_centroid_mode(self, X: pd.DataFrame) -> dict:
        """
        Calculates the mode (most frequent value) for each cluster's centroid.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data to calculate the centroids.

        Returns:
        -------
        dict
            A dictionary where the keys are cluster labels and the values are the mode for each cluster.
        """
        clusters = pd.Series(self.labels).unique()
        centroids = {}

        for cluster in clusters:
            cluster_data = X[pd.Series(self.labels) == cluster]
            centroids[cluster] = cluster_data.mode().iloc[0]

        return centroids

    def find_outliers(self, X: pd.DataFrame, outliers: Union[int, float], how: str = 'small_clusters') \
            -> Union[np.ndarray, List]:
        """
        Identifies outliers in the dataset based on a specified method.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data to find outliers in.

        outliers : float
            The percentage or number of outliers to identify.

        how : str, optional (default='small_clusters')
            The method to use for outlier detection. Options are:
            - 'small_clusters': Detects outliers based on small cluster sizes.
            - 'centeroids': Detects outliers based on Hamming distance from cluster centroids.
            - 'probability': Detects outliers based on probability distribution.

        Returns:
        -------
            A list where -1 represents outliers and other values represent cluster labels.
        """
        if outliers > 0:
            if how == 'small_clusters':
                if not isinstance(outliers, int):
                    raise "small_clusters requires a int for outliers"
                self.outliers = self.find_small_outlier_clusters(self.labels, outliers)
            elif how == 'centeroids':
                self.outliers = self.detect_outliers_hamming(X, outliers)
            elif how == 'probability':
                self.outliers = self.find_outliers_using_probability(X, outliers)
            else:
                raise ValueError("Invalid 'how' parameter")
            return self.outliers
        else:
            raise "There is no outliers to identify"

    def detect_outliers_hamming(self, X: pd.DataFrame, deviation_percent: float) -> Union[np.ndarray, List]:
        """
        Detects outliers based on Hamming distance from cluster centroids.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data to detect outliers in.

        deviation_percent : float
            The percentage of data points considered as outliers (e.g., 10 means 10% of the data).

        Returns:
        -------
        List
            A list of labels where -1 indicates an outlier, and 1 indicates a non-outlier.
        """
        n = len(X)
        outlier_flags = np.ones(n)
        distances = np.zeros(n)
        centroids = self.cluster_centroid_mode(X)

        for i, (label, row) in enumerate(zip(self.labels, X.values)):
            centroid = centroids[label]
            distances[i] = hamming(row, centroid)

        num_outliers = int(np.ceil(deviation_percent / 100 * n))
        outlier_indices = np.argsort(distances)[-num_outliers:]
        outlier_flags[outlier_indices] = -1

        return outlier_flags.tolist()

    def find_outliers_using_probability(self, df: pd.DataFrame, outlier_factor: float) -> Union[np.ndarray, List]:
        """
        Identifies outliers based on a probability distribution derived from the feature values.

        Parameters:
        ----------
        df : pd.DataFrame
            The input data to detect outliers in.

        outlier_factor : float
            The factor used to determine the number of outliers (e.g., 0.1 means 10% of the data).

        Returns:
        -------
        List
            A list where -1 indicates an outlier, and other values represent cluster labels.
        """
        outlier_factor = int(len(self.labels) * outlier_factor)
        unique_clusters = list(set(self.labels))
        clusters = np.array(self.labels)
        clusters_values = {}
        c_num = {}

        for c in unique_clusters:
            idx = np.where(clusters == c)[0]
            c_num[c] = len(idx)
            df_given_by_c = np.take(df, idx, 0)
            unique = list(set(i for j in df_given_by_c for i in j))
            clusters_values[c] = {v: np.count_nonzero(df_given_by_c == v) for v in unique}

        probabilities = []
        for i in range(len(df)):
            cl = clusters[i]
            probabilities.append(self.probability(df[i], clusters_values[cl], c_num[cl]))

        probabilities = np.array(probabilities)
        idx = np.argsort(probabilities)[:outlier_factor]
        for i in range(len(clusters)):
            if i in idx:
                clusters[i] = -1

        return list(clusters)

    @staticmethod
    def find_small_outlier_clusters(clusters: Union[np.ndarray, List], outlier_clusters_size: int = 6) \
            -> Union[np.ndarray, List]:
        """
        Identifies small clusters that are considered outliers.

        Parameters:
        ----------
        clusters : List
            The list of cluster labels.

        outlier_clusters_size : int, optional (default=6)
            The threshold size below which a cluster is considered an outlier.

        Returns:
        -------
        List
            A list where -1 indicates an outlier, and other values represent cluster labels.
        """
        counter = Counter(clusters)
        small_clusters = {k: counter[k] for k in counter if counter[k] <= outlier_clusters_size}
        sorted_clusters = dict(sorted(small_clusters.items(), key=lambda item: item[1]))

        outlier_clusters = []
        outlier_sum = 0
        outliers_count = 0
        for k, v in sorted_clusters.items():
            outlier_clusters.append(k)
            outlier_sum += v
            outliers_count += 1

        clusters = [i if i not in outlier_clusters else -1 for i in clusters]
        return clusters

    @staticmethod
    def create_regular_clusters_list(df: pd.DataFrame, results: List[List]) -> List:
        """
        Converts a list of clusters into a regular cluster assignment list.

        Parameters:
        ----------
        df : pd.DataFrame
            The input data to assign clusters to.

        results : List[List]
            A list of clusters where each sublist contains indices of samples in the same cluster.

        Returns:
        -------
        List
            A list of cluster assignments for each sample in the dataframe.
        """
        clusters = [0] * len(df)
        cluster_num = 1
        for c in results:
            for i in c:
                clusters[i] = cluster_num
            cluster_num += 1

        return clusters

    @staticmethod
    def probability(vector: List, values: dict, c_count: int) -> float:
        """
        Calculates the probability of a data point belonging to a cluster.

        Parameters:
        ----------
        vector : List
            The data vector for which the probability is calculated.

        values : dict
            The values (frequencies) associated with each feature for the cluster.

        c_count : int
            The number of samples in the cluster.

        Returns:
        -------
        float
            The probability of the data point belonging to the cluster.
        """
        P = 0
        for i in vector:
            P += values[i] / c_count
        return P

    @staticmethod
    def frame_to_vectors(df: pd.DataFrame) -> List[List]:
        """
        Converts a DataFrame to a list of vectors.

        Parameters:
        ----------
        df : pd.DataFrame
            The input data to convert.

        Returns:
        -------
        List[List]
            A list of vectors (rows) from the DataFrame.
        """
        return df.values.tolist()

    @staticmethod
    def frame_to_numpy(df: pd.DataFrame) -> np.ndarray:
        """
        Converts a DataFrame to a NumPy array.

        Parameters:
        ----------
        df : pd.DataFrame
            The input data to convert.

        Returns:
        -------
        np.ndarray
            The converted NumPy array.
        """
        return df.to_numpy()

    def select_outliers(self, threshold: int = 5) -> Union[np.ndarray, List]:
        """
        Identifies outliers based on the cluster size. If a cluster has fewer samples than the specified threshold,
        all its points are labeled as outliers.

        Parameters:
        ----------
        threshold : int, optional (default=5)
            The maximum number of data points a cluster can have before its points are considered outliers.

        Returns:
        -------
        List[int]
            A list of cluster labels where `-1` represents outliers, and other values represent valid clusters.
        """

        values, counts = np.unique(self.labels, return_counts=True)
        rare_values = values[counts <= threshold]
        result = self.labels.copy()
        for val in rare_values:
            result[result == val] = -1

        return result


