import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cblof import CBLOF
from sklearn.model_selection import ParameterGrid
import hdbscan
from evaluation.metrics import contrastive_outlier_score
from typing import Optional, Tuple

import warnings
warnings.simplefilter('ignore')


def HDBSCAN(df_ohe: np.ndarray) -> Optional[Tuple[hdbscan.HDBSCAN, dict, np.ndarray]]:
    """
    Perform HDBSCAN clustering with hyperparameter tuning using a grid search approach.

    This function tests multiple combinations of the 'min_cluster_size' and 'metric'
    hyperparameters for the HDBSCAN clustering algorithm, evaluating each combination
    using the contrastive outlier score. The best performing model and its associated
    parameters are returned, along with the final clustering labels.

    Parameters:
    ----------
    df_ohe : np.ndarray
        A numpy array containing the one-hot encoded data to be clustered.

    Returns:
    -------
    Optional[Tuple[hdbscan.HDBSCAN, dict, np.ndarray]]
        - The best HDBSCAN model after parameter tuning.
        - The best hyperparameters found during the grid search.
        - The cluster labels of the final model. Returns None if no valid model is found.
    """

    X = df_ohe.copy()

    best_score = -1
    best_model = None
    best_params = None

    for params in ParameterGrid({'min_cluster_size': [7, 10, 15, 20, 50], 'metric': ['hamming']}):
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(X)

        if len(set(labels)) > 1 and -1 in labels and np.sum(labels == -1) < len(df_ohe) and np.sum(labels != -1) < len(
                df_ohe):
            score = contrastive_outlier_score(X, labels, metric='hamming')

            if score > best_score:
                best_score = score
                best_params = params
                best_model = clusterer

    if best_params:
        db = hdbscan.HDBSCAN(**best_params)
        db_labels = db.fit_predict(X)
        return best_model, best_params, db_labels

    return None


def IsolationForest(df_ohe: np.ndarray, out_num: float) -> Optional[Tuple[IsolationForest, dict, np.ndarray]]:
    """
    Perform Isolation Forest anomaly detection with hyperparameter tuning using a grid search approach.

    This function tests multiple combinations of the 'n_estimators', 'max_samples', 'contamination',
    and 'max_features' hyperparameters for the Isolation Forest model, evaluating each combination
    using the contrastive outlier score. The best performing model and its associated parameters
    are returned, along with the final anomaly detection labels.

    Parameters:
    ----------
    df_ohe : np.ndarray
        A numpy array containing the one-hot encoded data to be processed by Isolation Forest.

    out_num : float
        The desired number (or proportion) of outliers. If out_num is greater than 1, it is treated as an
        absolute count of outliers. Otherwise, it is considered a proportion of the dataset.

    Returns:
    -------
    Optional[Tuple[IsolationForest, dict, np.ndarray]]
        - The best Isolation Forest model after parameter tuning.
        - The best hyperparameters found during the grid search.
        - The anomaly detection labels of the final model (1 for inliers, -1 for outliers).
        Returns None if no valid model is found.
    """

    X = df_ohe.copy()
    out_perc = out_num / len(df_ohe) if out_num > 1 else out_num

    best_model = None
    best_params = None
    best_iso_score = -np.inf

    for params in ParameterGrid({
        'n_estimators': [50, 100, 200, 300, 500, 700],
        'max_samples': [1.0, 0.5, 0.7, 0.8, 0.3],
        'contamination': [out_perc],
        'max_features': [1.0, 0.8]
    }):
        try:
            iso = IsolationForest(random_state=42, **params)
            iso.fit(X)
            preds = iso.predict(X)

            if np.sum(preds == -1) < len(df_ohe) and np.sum(preds != -1) < len(df_ohe):
                outlier_score = contrastive_outlier_score(X, preds, metric='hamming')

                if outlier_score > best_iso_score:
                    best_iso_score = outlier_score
                    best_model = iso
                    best_params = params
        except:
            continue

    if best_model is not None:
        labels = best_model.predict(X)
        return best_model, best_params, labels

    return None


def LOF(df_ohe: np.ndarray, out_num: float) -> Optional[Tuple[LocalOutlierFactor, dict, np.ndarray]]:
    """
    Perform Local Outlier Factor (LOF) anomaly detection with hyperparameter tuning using a grid search approach.

    This function tests multiple combinations of the 'n_neighbors', 'contamination', and 'metric' hyperparameters
    for the Local Outlier Factor model, evaluating each combination using the contrastive outlier score.
    The best performing model and its associated parameters are returned, along with the final anomaly detection labels.

    Parameters:
    ----------
    df_ohe : np.ndarray
        A numpy array containing the one-hot encoded data to be processed by the LOF algorithm.

    out_num : float
        The desired number (or proportion) of outliers. If out_num is greater than 1, it is treated as an
        absolute count of outliers. Otherwise, it is considered a proportion of the dataset.

    Returns:
    -------
    Optional[Tuple[LocalOutlierFactor, dict, np.ndarray]]
        - The best LOF model after parameter tuning.
        - The best hyperparameters found during the grid search.
        - The anomaly detection labels of the final model (1 for inliers, -1 for outliers).
        Returns None if no valid model is found.
    """

    X = df_ohe.copy()
    out_perc = out_num / len(df_ohe) if out_num > 1 else out_num

    best_lof_model = None
    best_params = None
    best_lof_score = -np.inf

    for params in ParameterGrid({
        'n_neighbors': [5, 10, 15, 20, 30],
        'contamination': [out_perc],
        'metric': ['hamming']
    }):
        try:
            lof = LocalOutlierFactor(novelty=False, **params)
            lof_preds = lof.fit_predict(X)

            if np.sum(lof_preds == -1) < len(df_ohe) and np.sum(lof_preds != -1) < len(df_ohe):
                score = contrastive_outlier_score(X, lof_preds, metric='hamming')

                if score > best_lof_score:
                    best_lof_score = score
                    best_lof_model = lof
                    best_params = params
        except:
            continue

    if best_lof_model is not None:
        labels = best_lof_model.fit_predict(X)
        return best_lof_model, best_params, labels

    return None


def CBLOF(df_ohe: np.ndarray, out_num: float) -> Optional[Tuple[dict, CBLOF, np.ndarray]]:
    """
    Perform CBLOF (Cluster-based Local Outlier Factor) anomaly detection with hyperparameter tuning using a grid search approach.

    This function tests multiple combinations of the 'n_clusters', 'contamination', 'alpha', and 'beta' hyperparameters
    for the CBLOF model, evaluating each combination using the contrastive outlier score.
    The best performing model and its associated parameters are returned, along with the final anomaly detection labels.

    Parameters:
    ----------
    df_ohe : np.ndarray
        A numpy array containing the one-hot encoded data to be processed by the CBLOF algorithm.

    out_num : float
        The desired number (or proportion) of outliers. If out_num is greater than 1, it is treated as an
        absolute count of outliers. Otherwise, it is considered a proportion of the dataset.

    Returns:
    -------
    Optional[Tuple[dict, CBLOF, np.ndarray]]
        - The best CBLOF model after parameter tuning.
        - The best hyperparameters found during the grid search.
        - The anomaly detection labels of the final model (1 for inliers, -1 for outliers).
        Returns None if no valid model is found.
    """

    X = df_ohe.copy()
    out_perc = out_num / len(df_ohe) if out_num > 1 else out_num

    best_cblof = None
    best_params = None
    best_cblof_score = -np.inf

    for params in ParameterGrid({
        'n_clusters': list(range(2, 30)),
        'contamination': [out_perc],
        'alpha': [0.6, 0.7, 0.8, 0.9],
        'beta': [2, 5]
    }):
        print(params)
        try:
            cblof = CBLOF(random_state=42, **params)
            cblof.fit(X)
            preds = cblof.labels_

            if np.sum(preds == -1) < len(df_ohe) and np.sum(preds != -1) < len(df_ohe):
                score = contrastive_outlier_score(X, preds, metric='hamming')

                if score > best_cblof_score:
                    best_cblof_score = score
                    best_cblof = cblof
                    best_params = params
        except:
            continue

    if best_cblof:
        labels = best_cblof.labels_
        return best_params, best_cblof, labels

    return None




if __name__ == "__main__":
    df = pd.read_csv(f'../data/synthetic/dataset_{0}.csv', sep=';')
    df_copy = df.copy()
    class_ = list(df_copy['class'])
    df_copy.drop('class', axis=1, inplace=True)
