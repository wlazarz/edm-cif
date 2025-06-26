import numpy as np
from numpy import log
from sklearn.metrics.cluster import (adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,
                                     mutual_info_score)
from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from collections import Counter
from typing import Dict, List, Union
from scipy.stats import entropy


def shannon_entropy(seq: List) -> float:
    """
    Computes Shannon entropy for a given sequence.

    Shannon entropy measures the uncertainty or unpredictability of a sequence.

    Parameters:
    ----------
    seq : List
        A list of elements (e.g., labels, values) for which the Shannon entropy is to be computed.

    Returns:
    -------
    float
        The Shannon entropy of the sequence, rounded to five decimal places. Returns 0 for an empty sequence.
    """
    n = len(seq)
    if n == 0:
        return 0
    counts = Counter(seq).values()
    return -sum((count / n) * log(count / n) for count in counts)


def shannon_for_full_set(labels: Union[np.ndarray, List[int]], class_values: Union[np.ndarray, List[int]]) -> float:
    """
    Computes Shannon entropy for the full set of labels across multiple clusters.

    This function calculates the average Shannon entropy across clusters, where entropy is computed for
    each cluster based on the distribution of class labels within that cluster.

    Parameters:
    ----------
    labels : List[int]
        A list of cluster labels for each data point.

    class_values : List[int]
        A list of class labels corresponding to each data point.

    Returns:
    -------
    float
        The average Shannon entropy across clusters, rounded to five decimal places. Returns 0 if no clusters are found.
    """
    cl_count = Counter(labels)
    entropy = 0

    for cl in cl_count:
        cluster_class_values = [class_values[i] for i in range(len(labels)) if labels[i] == cl]
        entropy += shannon_entropy(cluster_class_values)

    return float(round(entropy / len(cl_count), 5)) if cl_count else 0


def rand_index(clustering1: np.ndarray, clustering2: np.ndarray) -> float:
    """
    Compute the Adjusted Rand Index (ARI) between two clustering assignments.

    Parameters:
    ----------
    clustering1 : np.ndarray
        The first clustering labels as a numpy array.

    clustering2 : np.ndarray
        The second clustering labels as a numpy array.

    Returns:
    -------
    float
        The adjusted Rand index, rounded to five decimal places.
    """
    return float(round(adjusted_rand_score(clustering1, clustering2), 5))


def fowlkes_mallows_index(clustering1: np.ndarray, clustering2: np.ndarray) -> float:
    """
    Compute the Fowlkes-Mallows index between two clustering assignments.

    Parameters:
    ----------
    clustering1 : np.ndarray
        The first clustering labels as a numpy array.

    clustering2 : np.ndarray
        The second clustering labels as a numpy array.

    Returns:
    -------
    float
        The Fowlkes-Mallows index, rounded to five decimal places.
    """
    return float(round(fowlkes_mallows_score(clustering1, clustering2), 5))


def vi_index(clustering1: np.ndarray, clustering2: np.ndarray) -> float:
    """
    Compute the Variation of Information (VI) between two clustering assignments.

    Parameters:
    ----------
    clustering1 : np.ndarray
        The first clustering labels as a numpy array.

    clustering2 : np.ndarray
        The second clustering labels as a numpy array.

    Returns:
    -------
    float
        The Variation of Information score, rounded to five decimal places.
    """
    H_true = entropy(np.bincount(clustering1))
    H_pred = entropy(np.bincount(clustering2))

    I = mutual_info_score(clustering1, clustering2)
    VI = H_true + H_pred - 2 * I

    return float(round(VI, 5))


def nmi_index(clustering1: np.ndarray, clustering2: np.ndarray) -> float:
    """
    Compute the Normalized Mutual Information (NMI) between two clustering assignments.

    Parameters:
    ----------
    clustering1 : np.ndarray
        The first clustering labels as a numpy array.

    clustering2 : np.ndarray
        The second clustering labels as a numpy array.

    Returns:
    -------
    float
        The Normalized Mutual Information score, rounded to five decimal places.
    """
    return float(round(normalized_mutual_info_score(clustering1, clustering2), 5))


def label_distribution(clustering: Union[np.ndarray, List[int]], label: Union[np.ndarray, List[int]]) \
        -> Dict[str, Dict[str, int]]:
    """
    Compute the distribution of labels for each cluster in the clustering assignment.

    Parameters:
    ----------
    clustering : np.ndarray
        The clustering labels as a numpy array.

    label : np.ndarray
        The true labels for each data point.

    Returns:
    -------
    Dict[str, Dict[str, int]]
        A dictionary with the cluster label as the key and another dictionary mapping
        the true label to its count in that cluster.
    """
    counter_labels = {}
    for l in set(clustering):
        small_labels = label[clustering == l]
        counter_labels[str(l)] = {str(k): int(v) for k, v in dict(Counter(small_labels)).items()}

    return counter_labels


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy score between the true labels and predicted labels.

    Parameters:
    ----------
    y_true : np.ndarray
        The true labels.

    y_pred : np.ndarray
        The predicted labels.

    Returns:
    -------
    float
        The accuracy score, rounded to five decimal places.
    """
    return float(round(accuracy_score(y_true, y_pred), 5))


def precision(y_true: Union[np.ndarray, List[int]], y_pred: Union[np.ndarray, List[int]], avg: str = 'macro') -> float:
    """
    Compute the averaged Precision score.

    Precision is the ratio of correctly predicted positive observations to the total predicted positives.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    avg : str, optional (default='macro')
        The averaging method to compute the precision. Options are 'macro', 'micro', 'weighted', etc.

    Returns:
    -------
    float
        The precision score, rounded to five decimal places.
    """
    return float(round(precision_score(y_true, y_pred, average=avg), 5))


def recall(y_true: Union[np.ndarray, List[int]], y_pred: Union[np.ndarray, List[int]], avg: str = 'macro') -> float:
    """
    Compute the averaged Recall score.

    Recall is the ratio of correctly predicted positive observations to all observations in the actual class.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    avg : str, optional (default='macro')
        The averaging method to compute the recall. Options are 'macro', 'micro', 'weighted', etc.

    Returns:
    -------
    float
        The recall score, rounded to five decimal places.
    """
    return float(round(recall_score(y_true, y_pred, average=avg), 5))


def f1(y_true: List[int], y_pred: List[int], avg: str = 'macro') -> float:
    """
    Compute the averaged F1 score.

    The F1 score is the harmonic mean of precision and recall.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    avg : str, optional (default='macro')
        The averaging method to compute the F1 score. Options are 'macro', 'micro', 'weighted', etc.

    Returns:
    -------
    float
        The F1 score, rounded to five decimal places.
    """
    return float(round(f1_score(y_true, y_pred, average=avg), 5))


def matthews_corrcoef(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute the Matthews correlation coefficient (MCC) for multi-class classification.

    MCC is a measure of the quality of binary (or multi-class) classifications.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    Returns:
    -------
    float
        The Matthews correlation coefficient, rounded to five decimal places.
    """
    cm = confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)

    numerator = np.sum(np.diag(cm) * n - row_sum * col_sum)
    denominator = np.sqrt(np.sum(row_sum * (n - row_sum)) * np.sum(col_sum * (n - col_sum)))

    return float(round(numerator / (denominator + 1e-9), 5))


def cohens_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute Cohen's Kappa for multi-class classification.

    Cohen's Kappa is a statistic that measures inter-rater agreement for categorical items.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    Returns:
    -------
    float
        The Cohen's Kappa score, rounded to five decimal places.
    """
    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    po = np.trace(cm) / total
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2)
    return float(round((po - pe) / (1 - pe + 1e-9), 5))


def jaccard_index(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute the Jaccard Index (Macro-averaged).

    The Jaccard Index measures the similarity between two sets. In this case, it measures the similarity
    between the true and predicted class labels.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    Returns:
    -------
    float
        The Jaccard Index score, rounded to five decimal places.
    """
    return float(round(np.mean(np.array(y_true) == np.array(y_pred)), 5))


def hamming_loss(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute the Hamming Loss (Fraction of incorrect labels).

    The Hamming Loss is the fraction of labels that are incorrectly predicted.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    Returns:
    -------
    float
        The Hamming Loss score, rounded to five decimal places.
    """
    return float(round(np.mean(np.array(y_true) != np.array(y_pred)), 5))


def auc_roc(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute the Multi-class AUC-ROC score using a One-vs-Rest strategy.

    AUC-ROC is a performance measurement for classification problems,
    indicating the area under the receiver operating characteristic curve.
    It is computed for multi-class classification using the One-vs-Rest method.

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    Returns:
    -------
    float
        The AUC-ROC score, rounded to five decimal places.
    """
    class_names = np.unique(y_true)
    y_true_onehot = label_binarize(y_true, classes=class_names)
    y_pred_onehot = label_binarize(y_pred, classes=class_names)

    return float(round(roc_auc_score(y_true_onehot, y_pred_onehot, average="macro", multi_class="ovr"), 5))


def compare_methods(y_true: Union[np.ndarray, List[int]], y_pred: Union[np.ndarray, List[int]]) -> Dict[str, float]:
    """
    Compare various evaluation metrics for the classification performance.

    This function computes multiple evaluation metrics for the given true and predicted labels, including:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUC-ROC
    - Fowlkes-Mallows Index (FMI)
    - Adjusted Rand Index (ARI)
    - Variation of Information (VI)
    - Normalized Mutual Information (NMI)
    - Shannon Entropy
    - Cohen's Kappa
    - Jaccard Index
    - Matthews Correlation Coefficient (MCC)
    - Label Distribution

    Parameters:
    ----------
    y_true : List[int]
        The true class labels.

    y_pred : List[int]
        The predicted class labels.

    Returns:
    -------
    Dict[str, float]
        A dictionary where each key is a metric name and each value is the corresponding score.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1 REAL": f1_score(y_true, y_pred),
        "auc_roc": auc_roc(y_true, y_pred),
        "fmi": fowlkes_mallows_index(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "vi": vi_index(y_true, y_pred),
        "nmi": nmi_index(y_true, y_pred),
        "shannon": shannon_for_full_set(y_pred, y_true),
        "cohens_kappa": cohens_kappa(y_true, y_pred),
        "jaccard": jaccard_index(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "distribution": label_distribution(y_pred, y_true)
    }



