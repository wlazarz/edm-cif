import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from typing import Dict, List, Union, Optional


def compute_global_frequencies(X: np.ndarray) -> Dict:
    """
    Computes the frequency of each categorical value in the dataset.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array (or pandas DataFrame) containing categorical data.

    Returns:
    -------
    Dict
        A dictionary where keys are unique values and values are their occurrence counts.
    """
    frequencies = Counter(X.flatten())
    return dict(frequencies)


def compute_global_probabilities(X: np.ndarray) -> Dict:
    """
    Computes the probability of each categorical value in the dataset.

    The probability of each category is calculated as its frequency divided by
    the total number of elements in the dataset.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array (or pandas DataFrame) containing categorical data.

    Returns:
    -------
    Dict
        A dictionary where keys are unique values and values are their probabilities.
    """
    frequencies = compute_global_frequencies(X)
    total_values = X.size
    probabilities = {key: value / total_values for key, value in frequencies.items()}
    return probabilities


def eskin_distance(x: Union[np.ndarray, List[int]], y: Union[np.ndarray, List[int]],
                   unique_counts: Optional[List[int]] = None) -> float:
    """
    Computes the Eskin distance, which assigns higher weights to mismatches in attributes
    with many unique values.

    The Eskin distance between two vectors is a weighted sum of mismatches. Attributes
    with many unique values have higher weights.

    Parameters:
    ----------
    x : List[int]
        The first data point (list or numpy array of categorical values).

    y : List[int]
        The second data point (list or numpy array of categorical values).

    unique_counts : List[int]
        A list of the number of unique values for each feature in the dataset.

    Returns:
    -------
    float
        The Eskin distance between the two data points.
    """
    if unique_counts is None:
        m = len(x)
        return sum([0 if xi == yi else 10 / (10 + freq) for xi, yi, freq in zip(x, y, [1] * m)])

    distance = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            distance += 1 / (1 + unique_counts[i] ** 2)
        else:
            distance += 1
    return distance


def inverse_occurrence_frequency(x: List[int], y: List[int], global_frequencies: Dict[int, int]) -> float:
    """
    Computes the Inverse Occurrence Frequency (IOF) distance, which reduces the impact
    of differences in frequently occurring values.

    The IOF distance gives less weight to mismatches between common values and more weight
    to mismatches between rare values. The distance is computed using the logarithms of
    the frequency of values.

    Parameters:
    ----------
    x : List[int]
        The first data point (list or numpy array of categorical values).

    y : List[int]
        The second data point (list or numpy array of categorical values).

    global_frequencies : Dict[int, int]
        A dictionary where the keys are unique values and the values are their global occurrence counts.

    Returns:
    -------
    float
        The IOF distance between the two data points.
    """
    distance = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            freq = np.log(global_frequencies.get(x[i], 1)) + np.log(global_frequencies.get(y[i], 1))
            distance += 1 / freq
        else:
            distance += 1
    return distance


def lin_similarity(x: List[int], y: List[int], global_probabilities: Dict[int, float],
                   log_probabilities: Dict[int, float]) -> float:
    """
    Computes the Lin similarity based on information theory.

    The Lin similarity measures the similarity between two objects based on their shared attributes
    and the probabilities of those attributes. Higher probabilities lead to higher similarity.

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of categorical values).

    y : List[int]
        The second object (list or numpy array of categorical values).

    global_probabilities : Dict[int, float]
        A dictionary where keys are unique values and values are their probabilities.

    log_probabilities : Dict[int, float]
        A dictionary where keys are unique values and values are their log probabilities.

    Returns:
    -------
    float
        The Lin similarity value, higher values indicate more similarity.
    """
    similarity = 0
    denominator = 0
    for i in range(len(x)):
        a = x[i]
        b = y[i]

        if a == b:
            value = 2 * log_probabilities[a]
            similarity += value
            denominator += value
        else:
            similarity += 2 * np.log(global_probabilities[a] + global_probabilities[b])
            denominator += log_probabilities[a] + log_probabilities[b]

    return -similarity


def overlap_distance(x: Union[np.ndarray, List[int]], y: Union[np.ndarray, List[int]]) -> float:
    """
    Computes the Overlap distance (equivalent to Hamming distance for categorical data).

    The Overlap distance measures the proportion of differing attributes between two objects.
    The value ranges from 0 (identical) to 1 (completely different).

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of categorical values).

    y : List[int]
        The second object (list or numpy array of categorical values).

    Returns:
    -------
    float
        The Overlap distance (range [0, 1]).
    """
    sum_ = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            sum_ += 1
    return sum_ / len(x)


def overlap_similarity(x: Union[np.ndarray, List[int]], y: Union[np.ndarray, List[int]]) -> float:
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


def gower_distance(x: List[Union[int, float]], y: List[Union[int, float]], feature_types: List[str]) -> float:
    """
    Computes the Gower distance between two objects, considering both categorical and numerical features.

    The Gower distance is a measure of dissimilarity that handles both categorical and numerical data
    types and computes a distance between two objects. The distance ranges from 0 (identical) to 1 (completely different).

    Parameters:
    ----------
    x : List[Union[int, float]]
        The first object (list or numpy array of features).

    y : List[Union[int, float]]
        The second object (list or numpy array of features).

    feature_types : List[str]
        A list specifying whether each feature is 'categorical' or 'numerical'.

    Returns:
    -------
    float
        The Gower distance (range [0, 1]).
    """
    assert len(x) == len(y) == len(feature_types), "Input vectors must have the same length"

    total_distance = 0
    num_features = len(x)

    for i in range(num_features):
        if feature_types[i] == 'numerical':
            total_distance += abs(x[i] - y[i])
        elif feature_types[i] == 'categorical':
            total_distance += 1 if x[i] != y[i] else 0

    return total_distance / num_features


def hamming_distance(x: List[int], y: List[int]) -> int:
    """
    Computes the Hamming distance between two categorical objects.

    The Hamming distance counts the number of positions at which the corresponding elements are different.

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of categorical values).

    y : List[int]
        The second object (list or numpy array of categorical values).

    Returns:
    -------
    int
        The Hamming distance (number of differing positions).
    """
    sum_ = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            sum_ += 1
    return sum_


def dice_distance(x: List[int], y: List[int]) -> float:
    """
    Computes the Dice distance for categorical attributes.

    The Dice distance measures the similarity between two sets based on the number of common elements.
    The Dice coefficient is defined as the ratio of twice the number of common elements to the sum of
    the elements in both sets. The Dice distance is 1 minus this coefficient.

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of categorical values).

    y : List[int]
        The second object (list or numpy array of categorical values).

    Returns:
    -------
    float
        The Dice distance (range [0, 1]), where 0 means identical and 1 means completely different.
    """
    assert len(x) == len(y), "Input vectors must have the same length"

    matches = sum(1 for i in range(len(x)) if x[i] == y[i])
    total = len(x) * 2

    return 1 - (2 * matches / total)


def jaccard_coef(x: List[int], y: List[int]) -> float:
    """
    Computes the Jaccard coefficient for categorical attributes.

    The Jaccard coefficient measures the similarity between two sets as the ratio of the size of their
    intersection to the size of their union. The Jaccard distance is 1 minus the Jaccard coefficient.

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of categorical values).

    y : List[int]
        The second object (list or numpy array of categorical values).

    Returns:
    -------
    float
        The Jaccard coefficient (range [0, 1]), where 1 means identical and 0 means completely different.
    """
    sum_ = 0
    diff_ = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            sum_ += 1
        else:
            diff_ += 2
    return 1 - sum_ / (sum_ + diff_)


def create_dummy_features_frequencies(df: pd.DataFrame) -> List[int]:
    """
    Computes the frequency of each feature in a one-hot encoded DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        A one-hot encoded DataFrame where each column corresponds to a binary feature.

    Returns:
    -------
    List[int]
        A list of the sum of occurrences for each feature (i.e., frequency of each column being 1).
    """
    column_sums = df.sum()
    return column_sums.tolist()


def s2_distance(x: List[int], y: List[int], dummy_features_frequencies: List[int]) -> float:
    """
    Computes the S2 distance (Morlini & Zani), which considers co-occurrence of values in the dataset.

    The S2 distance measures the dissimilarity between two vectors, incorporating the frequency of co-occurrence
    of values in the dataset. This metric is based on information from one-hot encoded data.

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of binary values after one-hot encoding).

    y : List[int]
        The second object (list or numpy array of binary values after one-hot encoding).

    dummy_features_frequencies : List[int]
        A list of the frequency of appearance of each category in the dataset.

    Returns:
    -------
    float
        The S2 distance value (range [0, 1]), where 0 means identical and 1 means completely different.
    """
    counter = 0
    denominator = 0
    for i in range(len(x)):
        log_power = np.log(1 / (dummy_features_frequencies[i] ** 2))
        if x[i] == y[i] == 1:
            counter += log_power
            denominator += log_power
        elif (x[i] == 1 and y[i] == 0) or (y[i] == 1 and x[i] == 0):
            denominator += 2 * log_power

    return 1 - counter / denominator


def compute_unique_counts(X: pd.DataFrame) -> List[int]:
    """
    Computes the number of unique values for each feature in the dataset.

    Parameters:
    ----------
    X : pd.DataFrame
        A DataFrame containing categorical data.

    Returns:
    -------
    List[int]
        A list containing the number of unique values for each feature in the dataset.
    """
    result = []
    for c in X.columns:
        result.append(len(set(X[c])))
    return result


def get_metric(x: List[int], y: List[int], metric: str, dummy_features_frequencies: List[int],
               global_frequencies: Dict[int, int], global_probabilities: Dict[int, float],
               log_probabilities: Dict[int, float], unique_counts: List[int]) -> float:
    """
    Computes the specified metric between two data points based on their values.

    This function calculates various metrics such as the Eskin distance, IOF distance, Lin similarity,
    Overlap distance, Hamming distance, Dice distance, Jaccard coefficient, and S2 distance, based on the input metric.

    Parameters:
    ----------
    x : List[int]
        The first object (list or numpy array of categorical values).

    y : List[int]
        The second object (list or numpy array of categorical values).

    metric : str
        The name of the metric to compute ('eskin', 'iof', 'lin', 'overlap', 'hamming', 'dice', 'jaccard', 's2').

    dummy_features_frequencies : List[int]
        A list containing the frequency of appearance for each category in the dataset.

    global_frequencies : Dict[int, int]
        A dictionary where keys are unique values and values are their occurrence counts in the dataset.

    global_probabilities : Dict[int, float]
        A dictionary where keys are unique values and values are their probabilities.

    log_probabilities : Dict[int, float]
        A dictionary where keys are unique values and values are their log probabilities.

    unique_counts : List[int]
        A list of the number of unique values for each feature in the dataset.

    Returns:
    -------
    Tuple[float, str]
        A tuple containing the computed distance value
    """
    if metric == 'eskin':
        return eskin_distance(x, y, unique_counts)
    elif metric == 'iof':
        return inverse_occurrence_frequency(x, y, global_frequencies)
    elif metric == 'lin':
        return 1 - lin_similarity(x, y, global_probabilities, log_probabilities)
    elif metric == 'overlap':
        return overlap_distance(x, y)
    elif metric == 'hamming':
        return hamming_distance(x, y)
    elif metric == 'dice':
        return dice_distance(x, y)
    elif metric == 'jaccard':
        return 1 - jaccard_coef(x, y)
    elif metric == 's2':
        return s2_distance(x, y, dummy_features_frequencies)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def lin_similarity_vectorized(X: np.ndarray, global_probabilities: Dict[int, float]) -> np.ndarray:
    """
    Computes the pairwise Lin similarity for all objects in a dataset using NumPy and SciPy.

    The Lin similarity between two vectors is calculated based on information theory, with higher probabilities
    resulting in higher similarity. The pairwise similarities are computed for all objects in the dataset.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array where each row is an object (vector).

    global_probabilities : Dict[int, float]
        A dictionary where keys are unique values and values are their probabilities.

    Returns:
    -------
    np.ndarray
        A symmetric matrix of Lin similarity scores, where the element at position (i, j) represents the
        Lin similarity between objects i and j.
    """

    # Vectorizing the lookup for log probabilities
    prob_lookup = np.vectorize(lambda v: np.log(global_probabilities.get(v, 1e-6)))
    log_probs = prob_lookup(X)

    def lin_dist(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Lin similarity for two vectors."""
        same_mask = x == y
        log_px = log_probs[x]
        log_py = log_probs[y]

        numerator = 2 * np.sum(log_px[same_mask]) + 2 * np.sum(
            np.log(np.exp(log_px[~same_mask]) + np.exp(log_py[~same_mask])))
        denominator = np.sum(log_px) + np.sum(log_py)

        return numerator / (denominator + 1e-6)

    dist_matrix = squareform(pdist(X, metric=lin_dist))

    return dist_matrix


def s2_similarity_vectorized(data: np.ndarray, dummy_features_frequencies: List[int]) -> np.ndarray:
    """
    Computes the pairwise S2 distance matrix efficiently.

    The S2 distance (Morlini & Zani) takes into account the co-occurrence of values in the dataset. The metric
    is computed on one-hot encoded data, where higher frequencies of co-occurring values lead to smaller distances.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row is an object (after one-hot encoding).

    dummy_features_frequencies : List[int]
        A list containing the frequency of appearance for each category in the dataset.

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise S2 distances between all objects in the dataset.
    """
    n = data.shape[0]
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    log_power = np.log(1 / (np.array(dummy_features_frequencies) ** 2))

    for i in range(n):
        for j in range(i + 1, n):
            x, y = data[i], data[j]

            same_ones = (x == 1) & (y == 1)
            different = (x == 1) & (y == 0) | (x == 0) & (y == 1)

            counter = np.sum(log_power * same_ones)
            denominator = np.sum(log_power * same_ones) + np.sum(2 * log_power * different)

            result = counter / denominator
            distance_matrix[i, j] = result
            distance_matrix[j, i] = result

    return distance_matrix


def eskin_distance_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Eskin distance matrix efficiently. This is similar to the Overlap distance for categorical data.

    The Eskin distance assigns higher weights to mismatches in attributes with many unique values. It computes the
    pairwise distance between all objects in the dataset based on matching and non-matching feature values.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row is an object (vector of categorical features).

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise Eskin distances.
    """
    num_objects, num_features = data.shape
    distance_matrix = np.zeros((num_objects, num_objects))

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            matching_features = np.sum(data[i] == data[j])
            similarity = matching_features / num_features
            distance = 1 - similarity
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def iof_distance_vectorized(data: np.ndarray, global_frequencies: Dict[int, int]) -> np.ndarray:
    """
    Computes the pairwise Inverse Occurrence Frequency (IOF) distance matrix efficiently.

    The IOF distance reduces the impact of differences in frequently occurring values. For each feature,
    the frequency of its occurrence in the dataset is used to scale the differences between data points.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row is an object (vector of categorical features).

    global_frequencies : Dict[int, int]
        A dictionary mapping each unique value to its global frequency count in the dataset.

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise IOF distances.
    """
    n, d = data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    unique_values = np.unique(data)
    log_frequencies = {val: np.log(global_frequencies.get(val, 1)) for val in unique_values}
    log_freq_array = np.vectorize(log_frequencies.get)(data)

    for i, j in combinations(range(n), 2):
        x, y = data[i], data[j]
        same = x == y
        different = ~same
        freq = log_freq_array[i] + log_freq_array[j]
        distance = np.sum(same) + np.sum(different / freq)

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    return distance_matrix


def overlap_distance_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Overlap (Hamming) distance matrix efficiently.

    The Overlap distance measures how many attributes differ between pairs of objects. It is equivalent to the Hamming distance
    for categorical data, where the distance is the number of positions in which the features differ.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row is an object (vector of categorical features).

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise Overlap distances (Hamming distances).
    """
    n, d = data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    for i, j in combinations(range(n), 2):
        x, y = data[i], data[j]

        distance = np.sum(x != y) / d
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    return distance_matrix


def dice_distance_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Dice distance matrix efficiently.

    The Dice distance measures the dissimilarity between two sets as 1 minus the Dice coefficient. The Dice coefficient
    is defined as twice the number of common features divided by the sum of the features in both sets.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row is an object (vector of categorical features).

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise Dice distances.
    """
    n, d = data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    for i, j in combinations(range(n), 2):
        x, y = data[i], data[j]
        matches = np.sum(x == y)
        total = 2 * d

        distance = 1 - (2 * matches / total)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    return distance_matrix


def jaccard_coef_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Jaccard coefficient matrix efficiently.

    The Jaccard coefficient measures the similarity between two sets as the size of their intersection divided by the
    size of their union. The Jaccard distance is 1 minus the Jaccard coefficient.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row represents an object (vector of categorical features).

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise Jaccard coefficients, where each element represents the
        Jaccard coefficient between two objects.
    """
    n, d = data.shape
    similarity_matrix = np.zeros((n, n), dtype=np.float64)

    for i, j in combinations(range(n), 2):
        x, y = data[i], data[j]
        sum_ = np.sum(x == y)
        diff_ = np.sum(x != y) * 2

        similarity = 1 - (sum_ / (sum_ + diff_))
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

    return similarity_matrix


def hamming_distance_vectorized(data: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Hamming distance matrix efficiently.

    The Hamming distance counts the number of positions in which the corresponding elements of two objects differ.

    Parameters:
    ----------
    data : np.ndarray
        A 2D numpy array where each row represents an object (vector of categorical features).

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array with pairwise Hamming distances, where each element represents the
        number of differing positions between two objects.
    """
    n, d = data.shape
    distance_matrix = np.zeros((n, n), dtype=np.int32)

    for i, j in combinations(range(n), 2):
        x, y = data[i], data[j]

        distance = np.sum(x != y)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    return distance_matrix


def get_metric_vectorize(X: np.ndarray, metric: str, dummy_features_frequeces: List[int],
                         global_frequencies: Dict[int, int], global_probabilities: Dict[int, float]) -> np.ndarray:
    """
    Computes the specified metric between all pairs of objects in the dataset.

    This function calculates various metrics such as the Eskin distance, IOF distance, Lin similarity,
    Overlap distance, Hamming distance, Dice distance, Jaccard coefficient, and S2 distance, based on the input metric.

    Parameters:
    ----------
    X : np.ndarray
        A 2D numpy array where each row represents an object (vector of categorical features).

    metric : str
        The name of the metric to compute. Possible values: 'eskin', 'iof', 'lin', 'overlap', 'hamming',
        'dice', 'jaccard', 's2'.

    dummy_features_frequeces : List[int]
        A list containing the frequency of appearance for each category in the dataset.

    global_frequencies : Dict[int, int]
        A dictionary mapping each unique value to its global frequency count in the dataset.

    global_probabilities : Dict[int, float]
        A dictionary mapping each unique value to its probability in the dataset.

    Returns:
    -------
    np.ndarray
        A symmetric 2D numpy array representing the pairwise distances or similarities for the specified metric.
    """
    if metric == 'eskin':
        return eskin_distance_vectorized(X)
    elif metric == 'iof':
        return iof_distance_vectorized(X, global_frequencies)
    elif metric == 'lin':
        return 1 - lin_similarity_vectorized(X, global_probabilities)
    elif metric == 'overlap':
        return overlap_distance_vectorized(X)
    elif metric == 'hamming':
        return hamming_distance_vectorized(X)
    elif metric == 'dice':
        return dice_distance_vectorized(X)
    elif metric == 'jaccard':
        return 1 - jaccard_coef_vectorized(X)
    elif metric == 's2':
        return 1 - s2_similarity_vectorized(X, dummy_features_frequeces)
    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == '__main__':
    pass
