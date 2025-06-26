import pandas as pd
import random
from typing import Tuple
from utils.DatabaseUtils import SQLLiteUtils
from consts.const_variables import *
from utils.data_objects import Dataset
from dataclasses import asdict


def generate_categorical_dataset(
        n_samples: int = 500,
        n_features: int = 6,
        n_clusters: int = 4,
        noise_ratio: float = 0.15,
        category_range: Tuple[int, int] = (3, 8),
        anomaly_ratio: float = 0.1
) -> pd.DataFrame:
    """
    Generates a synthetic categorical dataset with a specified number of clusters and anomalies.

    Parameters:
    ----------
    n_samples : int, optional (default=500)
        The number of samples in the dataset.

    n_features : int, optional (default=6)
        The number of categorical features.

    n_clusters : int, optional (default=4)
        The number of clusters (excluding anomalies).

    noise_ratio : float, optional (default=0.15)
        The proportion of noisy values in the dataset.

    category_range : Tuple[int, int], optional (default=(3, 8))
        The range of the number of categories for each feature, defined as (min, max).

    anomaly_ratio : float, optional (default=0.1)
        The proportion of outliers (anomalies) in the dataset.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the generated categorical dataset with an additional `class` column:
        - `class`: The assigned cluster number (or `-1` for anomalies).
    """

    feature_categories = [random.randint(category_range[0], category_range[1]) for _ in range(n_features)]

    cluster_centers = []
    for _ in range(n_clusters):
        cluster_centers.append([random.randint(0, feature_categories[j] - 1) for j in range(n_features)])

    data = []
    labels = []

    for _ in range(int(n_samples * (1 - anomaly_ratio))):
        cluster_id = random.randint(0, n_clusters - 1)
        base_point = cluster_centers[cluster_id]

        noisy_point = [
            base_point[j] if random.random() > noise_ratio else random.randint(0, feature_categories[j] - 1)
            for j in range(n_features)
        ]

        data.append(noisy_point)
        labels.append(cluster_id)

    for _ in range(int(n_samples * anomaly_ratio)):
        anomaly_point = [random.randint(0, feature_categories[j] - 1) for j in range(n_features)]
        data.append(anomaly_point)
        labels.append(-1)

    column_names = [f'Feature_{i + 1}' for i in range(n_features)]

    df = pd.DataFrame(data, columns=column_names)
    df["class"] = labels

    return df


if __name__ == "__main__":

    db_session = SQLLiteUtils('../data/project_db.db')
    i=24
    for k in [2, 3, 5, 10]:
        for max_values in [5, 7, 9]:
            df = generate_categorical_dataset(n_samples=3000, n_features=5, n_clusters=k, category_range=(2, max_values),
                                              anomaly_ratio=0.15)
            df.to_csv(f'../data/synthetic/dataset_{i}.csv', index=False, sep=';')
            df_obj = Dataset(dataset_name=f'dataset_{i}',
                             path=f'data/synthetic/dataset_{i}.csv',
                             http_path=None,
                             labels = k,
                             rows=3000,
                             columns=5,
                             outliers_column='class',
                             is_synthetic=1,
                             outliers=0.15)
            db_session.insert_into_table_from_dictionary(datasets_table_name, asdict(df_obj))
            i += 1