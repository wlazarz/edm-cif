import copy
import datetime
import hashlib
import uuid
from dataclasses import asdict
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import Any

from evaluation.compare_clusterings import *
from evaluation.metrics import *


@dataclass
class OutlierDetectionProcess:
    algorithm: str
    dataset: str
    dataset_type: str

    split_mode: Optional[str]
    split_method: Optional[str]
    n_estimators: Optional[float]
    max_samples: Optional[float]
    min_cluster_size: Optional[float]
    contamination: Optional[float]
    n_neighbors: Optional[int]
    max_features: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]
    epsilon: Optional[float]
    k: Optional[int]
    m: Optional[int]
    theta: Optional[float]
    metric: Optional[str]
    ground_truth: int
    clustering: int
    outliers_threshold: Optional[int]
    time_sec: float
    task: Optional[str]

    detected_outliers_num: Optional[int] = None
    detected_outliers_perc: Optional[float] = None

    real_outliers_num: Optional[int] = None
    real_outliers_perc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    tp: Optional[int] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tpr: Optional[float] = None
    tnr: Optional[float] = None
    fpr: Optional[float] = None
    fnr: Optional[float] = None

    clusters: Optional[int] = None
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    dunn_index: Optional[float] = None
    cluster_entropy: Optional[float] = None

    fmi: Optional[float] = None
    ari: Optional[float] = None
    vi: Optional[float] = None
    nmi: Optional[float] = None
    shannon: Optional[float] = None

    process_id: Optional[str] = None
    distribution: Optional[str] = None

    def __post_init__(self):
        """Generate a process ID and set default task if not provided."""
        if not self.process_id:
            self.process_id = uuid.uuid4().hex
        if not self.task:
            self.task = 'outliers_detection'

    def supervised_clustering_metrics(self, y_real_no_outliers: np.ndarray, y_pred_no_outliers: np.ndarray):
        """
        Compute clustering performance metrics for supervised clustering.

        Args:
        y_real_no_outliers: Array of real labels without outliers.
        y_pred_no_outliers: Array of predicted labels without outliers.
        """
        if self.clusters > 1 and self.ground_truth == 1:
            self.fmi = round(float(fowlkes_mallows_index(y_real_no_outliers, y_pred_no_outliers)), 4)
            self.ari = round(float(adjusted_rand_score(y_real_no_outliers, y_pred_no_outliers)), 4)
            self.vi = round(float(vi_index(y_real_no_outliers, y_pred_no_outliers)), 4)
            self.nmi = round(float(nmi_index(y_real_no_outliers, y_pred_no_outliers)), 4)
            self.shannon = round(float(shannon_for_full_set(y_real_no_outliers, y_pred_no_outliers)), 4)

    def supervised_standard_metrics(self, y_real: np.ndarray, y_pred: np.ndarray):
        """
        Compute standard supervised metrics (precision, recall, accuracy, etc.) for outlier detection.

        Args:
        y_real: Array of true labels (ground truth).
        y_pred: Array of predicted labels (outlier detection results).
        """
        if self.ground_truth == 1:
            y_real_temp = y_real.copy()
            y_real_temp[y_real_temp != -1] = 1

            y_pred_temp = y_pred.copy()
            y_pred_temp[y_pred_temp != -1] = 1

            self.precision = round(float(precision(y_real_temp, y_pred_temp, 'macro')), 4)
            self.recall = round(float(recall(y_real_temp, y_pred_temp, 'macro')), 4)
            self.accuracy = round(float(accuracy(y_real_temp, y_pred_temp)), 4)
            self.tp = int(np.sum((y_real_temp == 1) & (y_pred_temp == 1)))
            self.tn = int(np.sum((y_real_temp == -1) & (y_pred_temp == -1)))
            self.fp = int(np.sum((y_real_temp == -1) & (y_pred_temp == 1)))
            self.fn = int(np.sum((y_real_temp == 1) & (y_pred_temp == -1)))

            self.tpr = round(float(self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0), 4)
            self.tnr = round(float(self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0), 4)
            self.fpr = round(float(self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0), 4)
            self.fnr = round(float(self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0), 4)

    def unsupervised_clustering_metrics(self, df_ohe: np.ndarray, df: np.ndarray, y_pred_no_outliers: np.ndarray):
        """
        Compute clustering performance metrics for unsupervised clustering.

        Args:
        df_ohe: One-hot encoded DataFrame of the dataset.
        df: Original DataFrame of the dataset.
        y_pred_no_outliers: Predicted labels with outliers removed.
        """
        if self.clustering == 1 and self.clusters > 1:
            self.silhouette_score = round(float(silhouette_score_hamming(df_ohe, y_pred_no_outliers)), 4)
            self.calinski_harabasz_score = round(float(calinski_harabasz_hamming(df_ohe, y_pred_no_outliers)), 4)
            self.davies_bouldin_score = round(float(davies_bouldin_hamming(df_ohe, y_pred_no_outliers)), 4)
            self.dunn_index = round(float(dunn_index_hamming(df_ohe, y_pred_no_outliers)), 4)
            self.cluster_entropy = round(float(cluster_entropy(df, y_pred_no_outliers)), 4)

    def count_metrics(self, df: pd.DataFrame, df_ohe: pd.DataFrame, y_real: np.ndarray, y_pred: np.ndarray):
        """
        Count metrics such as clusters, outlier detection, and clustering performance metrics.

        Args:
        df: The original dataset DataFrame.
        df_ohe: The one-hot encoded dataset DataFrame.
        y_real: The ground truth labels.
        y_pred: The predicted labels.
        """
        label_codes = {label: i if label not in ['-1', -1] else -1 for i, label in enumerate(set(y_pred))}
        y_pred = np.array([label_codes[i] for i in y_pred])

        if y_real is not None:
            label_codes = {label: i if label not in ['-1', -1] else -1 for i, label in enumerate(set(y_real))}
            y_real = np.array([label_codes[i] for i in y_real])

            mask = (y_real != -1) & (y_pred != -1)
            y_real_no_outliers = y_real[mask]
            y_pred_no_outliers = y_pred[mask]
        else:
            mask = (y_pred != -1)
            y_pred_no_outliers = y_pred[mask]
            y_real_no_outliers = None

        not_outlier_indicies = np.where(y_pred != -1)[0]

        self.clusters = len(set(y_pred_no_outliers))
        self.distribution = str(dict(Counter(Counter(int(x) for x in y_pred))))
        self.detected_outliers_num = int(np.sum(y_pred == -1))
        self.detected_outliers_perc = round(float(self.detected_outliers_num / len(y_pred)), 4)

        if self.ground_truth == 1:
            self.real_outliers_num = int(list(y_real).count(-1))
            self.real_outliers_perc = round(float(self.real_outliers_num / len(y_real)), 4)

        self.supervised_clustering_metrics(y_real_no_outliers, y_pred_no_outliers)
        self.supervised_standard_metrics(y_real, y_pred)
        self.unsupervised_clustering_metrics(df_ohe.loc[not_outlier_indicies].to_numpy(),
                                             df.loc[not_outlier_indicies].to_numpy(), y_pred[y_pred != -1])


@dataclass
class ContrastiveOutlierScore:
    """
    Class to calculate the Contrastive Outlier Score (COS) based on a given metric and strategy.

    The class allows the calculation of the Contrastive Outlier Score for a dataset using different distance metrics
    and aggregation strategies. The score measures how far data points are from the expected normal distribution of inliers
    given the outliers in the dataset.

    Attributes:
    ----------
    process_id : str
        Unique identifier for the process.
    metric : str
        The distance metric used for computing the score.
    strategy : str
        The aggregation strategy used to calculate the score.
    n : Optional[int]
        The number of closest neighbors to consider (only relevant for certain strategies).
    contrastive_outlier_score : Optional[float]
        The calculated contrastive outlier score for the dataset.
    """

    process_id: str
    metric: str
    strategy: str
    n: Optional[int] = None
    contrastive_outlier_score: Optional[float] = None

    def __init__(self, process_id: str, metric: str, strategy: str, n: Optional[int] = None):
        """
        Initialize the ContrastiveOutlierScore object with the provided parameters.

        Parameters:
        ----------
        process_id : str
            The unique process identifier.
        metric : str
            The distance metric used to compute the score.
        strategy : str
            The aggregation strategy for the outlier score.
        n : Optional[int], default=None
            The number of closest neighbors to consider (used in 'n_closest' strategy).
        """
        self.process_id = process_id
        self.metric = metric
        self.strategy = strategy
        self.n = n

    def calculate_score(self, df_ohe: np.ndarray, labels: np.ndarray) -> None:
        """
        Calculate the Contrastive Outlier Score for the given dataset.

        This method computes the Contrastive Outlier Score (COS) by applying the specified
        metric and strategy on the one-hot encoded data and the corresponding labels.

        Parameters:
        ----------
        df_ohe : np.ndarray
            The one-hot encoded dataset as a numpy array.
        labels : np.ndarray
            The labels corresponding to the dataset, with outliers marked as -1.

        Updates:
        -------
        self.contrastive_outlier_score : float
            The calculated contrastive outlier score for the dataset, rounded to 4 decimal places.
        """
        self.contrastive_outlier_score = round(float(contrastive_outlier_score(df_ohe, labels, self.metric,
                                                                               self.strategy, self.n)), 4)


@dataclass
class ContrastiveOutlier:
    """
    Class to calculate and store contrastive outlier scores for a dataset based on different configurations.

    The class allows the calculation of contrastive outlier scores for various configurations of metrics and strategies.
    The results are stored in a list of `ContrastiveOutlierScore` objects, which are further stored in a database.

    Attributes:
    ----------
    process_id : str
        Unique identifier for the process.
    contrastive_outlier_list : Optional[List[ContrastiveOutlierScore]]
        A list of ContrastiveOutlierScore objects, each representing a specific configuration's result.
    """

    process_id: str
    contrastive_outlier_list: Optional[List['ContrastiveOutlierScore']] = None

    def calculate_configurations(self, df_ohe: np.ndarray, labels: np.ndarray) -> None:
        """
        Calculate contrastive outlier scores for different configurations (metric and strategy combinations).

        This method iterates over different metrics and strategies to compute the contrastive outlier scores.
        The configurations depend on the number of unique labels in the dataset. If there are exactly 2 labels,
        it computes scores for different `n` values in the range [10%, 95%]. If there are more than 2 labels,
        it computes scores based on the minimum and maximum class counts.

        Parameters:
        ----------
        df_ohe : np.ndarray
            The one-hot encoded dataset as a numpy array.

        labels : np.ndarray
            The labels of the dataset, with outliers marked as -1.
        """
        if len(set(labels)) == 2:
            n_range = [int(round(len(labels) * perc / 100)) for perc in range(10, 95, 5)]
        elif len(set(labels)) > 2:
            values, counts = np.unique(labels[labels != -1], return_counts=True)
            min_count = counts.min()
            max_count = counts.max()
            n_range = [int(i) for i in set(np.round(np.linspace(min(min_count, 10), max_count // 2, 18)))]
        else:
            raise "There is only one unique label in the dataset."

        if not self.contrastive_outlier_list:
            self.contrastive_outlier_list = []

            for metric in ['hamming', 'jaccard', 'cosine']:
                for strategy in ['average', 'median', 'n_closest']:
                    if strategy == 'n_closest':
                        for n in n_range:
                            score_obj = ContrastiveOutlierScore(process_id=self.process_id, metric=metric,
                                                                strategy=strategy, n=n)
                            score_obj.calculate_score(df_ohe, labels)
                            self.contrastive_outlier_list.append(score_obj)
                    else:
                        score_obj = ContrastiveOutlierScore(process_id=self.process_id, metric=metric,
                                                            strategy=strategy)
                        score_obj.calculate_score(df_ohe, labels)
                        self.contrastive_outlier_list.append(score_obj)

    def insert_to_db(self, db_session, table_name: str = 'contrastive_outlier_score') -> None:
        """
        Insert the contrastive outlier scores into the specified database table.

        This method inserts each contrastive outlier score in the `contrastive_outlier_list` into the database.

        Parameters:
        ----------
        db_session : object
            The database session object used to interact with the database.

        table_name : str, optional (default='contrastive_outlier_score')
            The name of the table where the data will be inserted.
        """
        for score_obj in self.contrastive_outlier_list:
            db_session.insert_into_table_from_dictionary(table_name, asdict(score_obj))



@dataclass
class LabelingProcess:
    """
    Class representing a labeling process for a dataset.

    This class encapsulates the settings for labeling a dataset, including the method used, algorithm, input samples,
    and optional parameters. It also ensures that a unique process ID is generated and stored for each instance.

    Attributes:
    ----------
    method : str
        The labeling method used.

    algorithm : Optional[str], optional
        The algorithm used for labeling, if any.

    dataset : str
        The dataset being labeled.

    input_samples : int
        The number of input samples.

    input_samples_method : str
        The method used for selecting input samples.

    iters : Optional[int], optional
        The number of iterations for the process, if applicable.

    strategy : Optional[str], optional
        The strategy for labeling, if applicable.

    metric : Optional[str], optional
        The metric used for evaluation, if any.

    params : Optional[Any], optional
        Additional parameters for the labeling process.

    process_id : Optional[str], optional
        A unique identifier for the process.

    task : Optional[str], default='labeling'
        The task being performed, which is 'labeling' by default.
    """

    method: str
    algorithm: Optional[str]
    dataset: str
    input_samples: int
    input_samples_method: str
    iters: Optional[int] = None
    strategy: Optional[str] = None
    metric: Optional[str] = None
    params: Optional[Any] = None

    process_id: Optional[str] = None
    task: Optional[str] = 'labeling'

    def __post_init__(self):
        """
        Initialize the process by setting a unique process ID if it's not provided,
        and ensure the `params` attribute is properly formatted.
        """
        if not self.process_id:
            self.process_id = uuid.uuid4().hex
        if self.params and not isinstance(self.params, str):
            self.params = str(self.params)


@dataclass
class ClassEvaluationProcess:
    """
    Class representing a class evaluation process for a dataset.

    This class holds the necessary attributes for evaluating a classification process, including dataset details,
    algorithm parameters, and performance metrics. It ensures a unique process ID is generated for each instance.

    Attributes:
    ----------
    dataset : str
        The dataset being evaluated.

    non_error_ratio : Optional[float], optional
        The non-error ratio metric for the evaluation.

    algorithm : Optional[str], optional
        The algorithm used for the class evaluation.

    alg_param_1 : Optional[float], optional
        The first algorithm parameter.

    alg_param_2 : Optional[float], optional
        The second algorithm parameter.

    params : Optional[Any], optional
        Additional parameters for the class evaluation process.

    process_id : Optional[str], optional
        A unique identifier for the process.

    task : Optional[str], default='class_evaluation'
        The task being performed, which is 'class_evaluation' by default.
    """

    dataset: str
    non_error_ratio: Optional[float] = None
    algorithm: Optional[str] = None
    alg_param_1: Optional[float] = None
    alg_param_2: Optional[float] = None
    params: Optional[Any] = None

    process_id: Optional[str] = None
    task: Optional[str] = 'class_evaluation'

    def __post_init__(self):
        """
        Initialize the process by setting a unique process ID if it's not provided,
        and ensure the `params` attribute is properly formatted.
        """
        if not self.process_id:
            self.process_id = uuid.uuid4().hex
        if self.params and not isinstance(self.params, str):
            self.params = str(self.params)


@dataclass
class ComparisonMetrics:
    """
    Class to compute and store comparison metrics between true and predicted labels.

    The class supports various evaluation metrics such as accuracy, precision, recall, F1-score, and others.
    It calculates clustering performance metrics such as the Adjusted Rand Index (ARI), Fowlkes-Mallows Index (FMI),
    and others for both supervised and unsupervised tasks.

    Attributes:
    ----------
    process_id : str
        Unique identifier for the process.
    clusters1 : Optional[int], optional
        Number of clusters in the true labels.
    clusters2 : Optional[int], optional
        Number of clusters in the predicted labels.
    accuracy : Optional[float], optional
        Accuracy score between true and predicted labels.
    macro_precision : Optional[float], optional
        Macro-averaged precision score.
    micro_precision : Optional[float], optional
        Micro-averaged precision score.
    macro_recall : Optional[float], optional
        Macro-averaged recall score.
    micro_recall : Optional[float], optional
        Micro-averaged recall score.
    macro_f1 : Optional[float], optional
        Macro-averaged F1-score.
    micro_f1 : Optional[float], optional
        Micro-averaged F1-score.
    auc_roc : Optional[float], optional
        Area under the ROC curve (AUC-ROC).
    fmi : Optional[float], optional
        Fowlkes-Mallows Index (FMI).
    ari : Optional[float], optional
        Adjusted Rand Index (ARI).
    vi : Optional[float], optional
        Variation of Information (VI).
    nmi : Optional[float], optional
        Normalized Mutual Information (NMI).
    shannon : Optional[float], optional
        Shannon entropy.
    cohens_kappa : Optional[float], optional
        Cohen's Kappa score.
    jaccard : Optional[float], optional
        Jaccard index score.
    matthews_corrcoef : Optional[float], optional
        Matthews correlation coefficient.
    distribution : Optional[str], optional
        Distribution of the predicted labels compared to the true labels.
    task : Optional[str], default='labeling'
        The task being performed (e.g., 'labeling', 'clustering').

    Methods:
    -------
    encode_y(y_true_, y_pred_):
        Encodes true and predicted labels using LabelEncoder.
    make_comparison(y_true_, y_pred_):
        Computes various comparison metrics between true and predicted labels.
    """

    process_id: str
    clusters1: Optional[int] = None
    clusters2: Optional[int] = None
    accuracy: Optional[float] = None
    macro_precision: Optional[float] = None
    micro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    micro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    micro_f1: Optional[float] = None
    auc_roc: Optional[float] = None
    fmi: Optional[float] = None
    ari: Optional[float] = None
    vi: Optional[float] = None
    nmi: Optional[float] = None
    shannon: Optional[float] = None
    cohens_kappa: Optional[float] = None
    jaccard: Optional[float] = None
    matthews_corrcoef: Optional[float] = None
    distribution: Optional[str] = None
    task: Optional[str] = 'labeling'

    def __post_init__(self):
        """
        Initializes the process by setting a unique process ID if not provided,
        and sets the task to 'labeling' by default.
        """
        if not self.process_id:
            self.process_id = uuid.uuid4().hex

    def encode_y(self, y_true_: List[int], y_pred_: List[int]) -> tuple:
        """
        Encodes the true and predicted labels using LabelEncoder.

        Parameters:
        ----------
        y_true_ : List[int]
            The true labels for the dataset.

        y_pred_ : List[int]
            The predicted labels for the dataset.

        Returns:
        -------
        tuple
            A tuple containing the encoded true and predicted labels.
        """
        all_unique_values = list(set(y_true_) | set(y_pred_))
        encoder = LabelEncoder()
        encoder.fit(all_unique_values)
        return encoder.transform(y_true_), encoder.transform(y_pred_)

    def make_comparison(self, y_true_: List[int], y_pred_: List[int]) -> None:
        """
        Computes various comparison metrics between the true and predicted labels.

        The metrics include accuracy, precision, recall, F1-score, and clustering evaluation metrics
        such as Fowlkes-Mallows Index, Adjusted Rand Index, and others.

        Parameters:
        ----------
        y_true_ : List[int]
            The true labels for the dataset.

        y_pred_ : List[int]
            The predicted labels for the dataset.
        """
        if not self.distribution:
            self.distribution = str(label_distribution(y_pred_, y_true_))

        y_true, y_pred = self.encode_y(y_true_, y_pred_)

        if not self.clusters1:
            self.clusters1 = len(set(y_true))
        if not self.clusters2:
            self.clusters2 = len(set(y_pred))

        if self.task != 'clustering':
            if not self.accuracy:
                self.accuracy = accuracy(y_true, y_pred)
            if not self.macro_precision:
                self.macro_precision = precision(y_true, y_pred, 'macro')
            if not self.micro_precision:
                self.micro_precision = precision(y_true, y_pred, 'micro')
            if not self.macro_recall:
                self.macro_recall = recall(y_true, y_pred, 'macro')
            if not self.micro_recall:
                self.micro_recall = recall(y_true, y_pred, 'micro')
            if not self.macro_f1:
                self.macro_f1 = f1(y_true, y_pred, 'macro')
            if not self.micro_f1:
                self.micro_f1 = f1(y_true, y_pred, 'micro')

        if not self.fmi:
            self.fmi = fowlkes_mallows_index(y_true, y_pred)
        if not self.ari:
            self.ari = adjusted_rand_score(y_true, y_pred)
        if not self.vi:
            self.vi = vi_index(y_true, y_pred)
        if not self.nmi:
            self.nmi = nmi_index(y_true, y_pred)
        if not self.shannon:
            self.shannon = shannon_for_full_set(y_pred, y_true)
        if not self.cohens_kappa:
            self.cohens_kappa = cohens_kappa(y_true, y_pred)
        if not self.jaccard:
            self.jaccard = jaccard_index(y_true, y_pred)
        if not self.matthews_corrcoef:
            self.matthews_corrcoef = matthews_corrcoef(y_true, y_pred)

@dataclass
class CutPointsComparison(ComparisonMetrics):
    """
    A subclass of `ComparisonMetrics` that represents a comparison of cut points
    in a dataset and includes additional attributes for cut point value and the
    number of elements associated with that cut point.

    Attributes:
    ----------
    cut_point : float
        The value of the cut point being evaluated.

    n_elements : int
        The number of elements associated with the cut point.
    """

    cut_point: float = 0
    n_elements: int = 0


@dataclass
class EvaluationMetricsWithMeasure:
    """
    Class to compute and store evaluation metrics with additional clustering measures
    for dataset comparison. It includes metrics like silhouette score, Dunn index,
    Davies-Bouldin score, and Calinski-Harabasz score.

    Attributes:
    ----------
    process_id : str
        Unique identifier for the process.

    metric : str
        The evaluation metric being used.

    silhouette_score : Optional[float], optional
        Silhouette score for clustering evaluation.

    dunn_index : Optional[float], optional
        Dunn index for clustering evaluation.

    davies_bouldin_score : Optional[float], optional
        Davies-Bouldin score for clustering evaluation.

    calinski_harabasz_score : Optional[float], optional
        Calinski-Harabasz score for clustering evaluation.

    Methods:
    -------
    fix_bool(df):
        Converts boolean columns in a DataFrame to integers (0 or 1).

    count_metrics(df_, clusters_, con_matrix):
        Computes various evaluation metrics for clustering performance.
    """

    process_id: str
    metric: str
    silhouette_score: Optional[float] = None
    dunn_index: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None

    @staticmethod
    def fix_bool(df: np.ndarray) -> np.ndarray:
        """
        Converts boolean columns in the DataFrame to integers (0 or 1).

        Parameters:
        ----------
        df : np.ndarray
            The dataset to check for boolean columns.

        Returns:
        -------
        np.ndarray
            The DataFrame with boolean columns converted to integers.
        """
        bool_columns = df.dtype == bool
        if bool_columns:
            df = df.astype(int)
        return df

    def count_metrics(self, df_: np.ndarray, clusters_: np.ndarray, con_matrix: np.ndarray) -> None:
        """
        Computes and stores evaluation metrics for clustering, including Dunn index,
        silhouette score, Davies-Bouldin score, and Calinski-Harabasz score.

        Parameters:
        ----------
        df_ : np.ndarray
            The dataset to use for metric computation.

        clusters_ : np.ndarray
            The clustering labels corresponding to the dataset.

        con_matrix : np.ndarray
            The distance or similarity matrix for the dataset.
        """
        if clusters_ is not None and clusters_.size > 0 and not isinstance(clusters_, np.ndarray):
            clusters = np.array(clusters_)
        else:
            clusters = copy.deepcopy(clusters_)

        if df_ is not None and df_.size > 0 and not isinstance(df_, np.ndarray):
            df = self.fix_bool(np.array(df_))
        else:
            df = copy.deepcopy(df_)

        if not self.dunn_index:
            self.dunn_index = dunn_user_metric(clusters, con_matrix)
        if not self.silhouette_score:
            self.silhouette_score = silhouette_score_user_metric(clusters, con_matrix)
        if not self.calinski_harabasz_score:
            self.calinski_harabasz_score = calinski_harabasz_user_metric(df, clusters, con_matrix)
        if not self.davies_bouldin_score:
            self.davies_bouldin_score = davies_bouldin_user_metric(clusters, con_matrix)


@dataclass
class EvaluationMetrics:
    """
    Class to compute and store various evaluation metrics for clustering performance.

    The class includes methods to compute clustering evaluation metrics such as silhouette score,
    Calinski-Harabasz score, Davies-Bouldin score, Dunn index, cluster entropy, and several custom
    metrics like M1, M2, M3, and M4. These metrics are used to evaluate the quality of clustering solutions.

    Attributes:
    ----------
    process_id : str
        Unique identifier for the process.

    silhouette_score : Optional[float], optional
        The silhouette score for clustering evaluation.

    silhouette_score_hamming : Optional[float], optional
        The silhouette score with Hamming distance for clustering evaluation.

    calinski_harabasz_score : Optional[float], optional
        The Calinski-Harabasz score for clustering evaluation.

    calinski_harabasz_score_hamming : Optional[float], optional
        The Calinski-Harabasz score with Hamming distance for clustering evaluation.

    davies_bouldin_score : Optional[float], optional
        The Davies-Bouldin score for clustering evaluation.

    davies_bouldin_score_hamming : Optional[float], optional
        The Davies-Bouldin score with Hamming distance for clustering evaluation.

    dunn_index : Optional[float], optional
        The Dunn index for clustering evaluation.

    dunn_index_hamming : Optional[float], optional
        The Dunn index with Hamming distance for clustering evaluation.

    cluster_entropy : Optional[float], optional
        The entropy of the clustering distribution.

    cluster_inconsistency : Optional[float], optional
        The inconsistency of the clustering.

    cluster_separation : Optional[float], optional
        The separation of the clusters.

    profit : Optional[float], optional
        The profit score for clustering evaluation.

    hamming_ratio : Optional[float], optional
        The Hamming ratio for clustering evaluation.

    m1 : Optional[float], optional
        The M1 score for clustering evaluation.

    m2 : Optional[float], optional
        The M2 score for clustering evaluation.

    m3 : Optional[float], optional
        The M3 score for clustering evaluation.

    m4 : Optional[float], optional
        The M4 score for clustering evaluation.

    Methods:
    -------
    fix_bool(df):
        Converts boolean columns in the DataFrame to integers (0 or 1).

    count_metrics(df_, df_ohe_, clusters_):
        Computes various evaluation metrics for clustering performance.
    """

    process_id: str
    silhouette_score: Optional[float] = None
    silhouette_score_hamming: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    calinski_harabasz_score_hamming: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    davies_bouldin_score_hamming: Optional[float] = None
    dunn_index: Optional[float] = None
    dunn_index_hamming: Optional[float] = None
    cluster_entropy: Optional[float] = None
    cluster_inconsistency: Optional[float] = None
    cluster_separation: Optional[float] = None
    profit: Optional[float] = None
    hamming_ratio: Optional[float] = None
    m1: Optional[float] = None
    m2: Optional[float] = None
    m3: Optional[float] = None
    m4: Optional[float] = None

    @staticmethod
    def fix_bool(df: np.ndarray) -> np.ndarray:
        """
        Converts boolean columns in a DataFrame to integers (0 or 1).

        Parameters:
        ----------
        df : np.ndarray
            The dataset to check for boolean columns.

        Returns:
        -------
        np.ndarray
            The DataFrame with boolean columns converted to integers.
        """
        bool_columns = df.dtype == bool
        if bool_columns:
            df = df.astype(int)

        return df

    def count_metrics(self, df_: np.ndarray, df_ohe_: np.ndarray, clusters_: np.ndarray) -> None:
        """
        Computes various evaluation metrics for clustering performance. This includes metrics such as
        silhouette score, Davies-Bouldin score, Dunn index, and several custom metrics (M1, M2, M3, M4).

        Parameters:
        ----------
        df_ : np.ndarray
            The dataset to use for metric computation.

        df_ohe_ : np.ndarray
            The one-hot encoded dataset for computing metrics like Hamming distance.

        clusters_ : np.ndarray
            The clustering labels corresponding to the dataset.
        """

        if clusters_ is not None and clusters_.size > 0 and not isinstance(clusters_, np.ndarray):
            clusters = np.array(clusters_)
        else:
            clusters = copy.deepcopy(clusters_)

        if df_ohe_ is not None and df_ohe_.size > 0 and not isinstance(df_ohe_, np.ndarray):
            df_ohe = self.fix_bool(np.array(df_ohe_))
        else:
            df_ohe = copy.deepcopy(df_ohe_)

        if df_ is not None and df_.size > 0 and not isinstance(df_, np.ndarray):
            df = self.fix_bool(np.array(df_))
        else:
            df = copy.deepcopy(df_)

        if not self.silhouette_score:
            self.silhouette_score = silhouette_score_calc(df_ohe, clusters)
        if not self.silhouette_score_hamming:
            self.silhouette_score_hamming = silhouette_score_hamming(df_ohe, clusters)
        if not self.calinski_harabasz_score:
            self.calinski_harabasz_score = calinski_harabasz_score_calc(df_ohe, clusters)
        if not self.calinski_harabasz_score_hamming:
            self.calinski_harabasz_score_hamming = calinski_harabasz_hamming(df_ohe, clusters)
        if not self.davies_bouldin_score:
            self.davies_bouldin_score = davies_bouldin_score_calc(df_ohe, clusters)
        if not self.davies_bouldin_score_hamming:
            self.davies_bouldin_score_hamming = davies_bouldin_hamming(df_ohe, clusters)
        if not self.dunn_index:
            self.dunn_index = dunn_index(df_ohe, clusters)
        if not self.dunn_index_hamming:
            self.dunn_index_hamming = dunn_index_hamming(df_ohe, clusters)
        if not self.cluster_entropy:
            self.cluster_entropy = cluster_entropy(df, clusters)
        if not self.cluster_inconsistency:
            self.cluster_inconsistency = cluster_inconsistency(df_ohe, clusters)
        if not self.cluster_separation:
            self.cluster_separation = cluster_separation(df_ohe, clusters)
        if not self.profit:
            self.profit = profit(df, clusters)
        if not self.hamming_ratio:
            self.hamming_ratio = hamming_ratio(df_ohe, clusters)
        if not self.m1:
            self.m1 = M1(df, clusters)
        if not self.m2:
            self.m2 = M2(df, clusters)
        if not self.m3:
            self.m3, self.m4 = M3(df, clusters)

@dataclass
class Dataset:
    """
    Class to represent a dataset with attributes for storing metadata, such as dataset name, path,
    extension, labels, rows, columns, and various other dataset characteristics. The class provides
    methods to store and process dataset information.

    Attributes:
    ----------
    dataset_name : str
        Name of the dataset.

    path : Union[Path, bytes, str]
        The path where the dataset is stored.

    http_path : Optional[str], optional
        The HTTP path (URL) where the dataset is accessible, if available.

    dataset_id : Optional[str], optional
        A unique identifier for the dataset, automatically generated from the dataset name.

    extension : Optional[str], optional
        File extension of the dataset file.

    labels : Optional[int], optional
        The number of unique labels (for classification tasks).

    rows : Optional[int], optional
        The number of rows in the dataset.

    columns : Optional[int], optional
        The number of columns in the dataset.

    numerical_column_names : Optional[str], optional
        A comma-separated string of numerical column names.

    numerical_columns_number : Optional[int], optional
        The number of numerical columns in the dataset.

    categorical_column_names : Optional[str], optional
        A comma-separated string of categorical column names.

    categorical_columns_number : Optional[int], optional
        The number of categorical columns in the dataset.

    outliers_column : Optional[str], optional
        The name of the column that marks outliers.

    is_synthetic : Optional[int], default=False
        Flag indicating whether the dataset is synthetic (1 for synthetic, 0 for real).

    outliers : Optional[float], default=0
        The percentage of outliers in the dataset.

    currdate : Optional[datetime], default=datetime.datetime.now()
        The current date and time of dataset creation or processing.

    Methods:
    -------
    __post_init__():
        Initializes the `dataset_id` and `extension` attributes if not provided.

    dataset_info(df: pd.DataFrame, class_col: Optional[str] = None, outlier_mark: Optional[Any] = None):
        Populates dataset metadata based on the provided DataFrame, including label counts, rows, columns,
        and categorical/numerical column names. Also handles outlier marking.

    prepare_data_and_save(df: pd.DataFrame, dest_path: Union[Path, bytes, str], class_col: Optional[str] = None,
                          outlier_mark: Optional[Any] = None):
        Prepares the dataset (renaming columns, handling outliers) and saves it to the specified destination path.
    """

    dataset_name: str
    path: Union[Path, bytes, str]
    http_path: Optional[str]

    dataset_id: Optional[str] = None
    extension: Optional[str] = None
    labels: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    numerical_column_names: Optional[str] = None
    numerical_columns_number: Optional[int] = None
    categorical_column_names: Optional[str] = None
    categorical_columns_number: Optional[int] = None
    outliers_column: Optional[str] = None
    is_synthetic: Optional[int] = False
    outliers: Optional[float] = 0
    currdate: Optional[datetime] = datetime.datetime.now()

    def __post_init__(self):
        """
        Initializes the `dataset_id` attribute by generating a hash of the `dataset_name`,
        and sets the file `extension` if not already provided.
        """
        if not self.dataset_id:
            self.dataset_id = hashlib.md5(self.dataset_name.encode()).hexdigest()
        if not self.extension:
            self.extension = Path(self.path).suffix

    def dataset_info(self, df: pd.DataFrame, class_col: Optional[str] = None,
                     outlier_mark: Optional[Any] = None) -> None:
        """
        Populates the dataset metadata, including the number of labels, rows, columns,
        and the names of numerical and categorical columns. Also handles the marking of outliers.

        Parameters:
        ----------
        df : pd.DataFrame
            The dataset in DataFrame format.

        class_col : Optional[str], optional
            The column name containing the class labels for classification tasks.

        outlier_mark : Optional[Any], optional
            The value used to identify outliers in the dataset.
        """
        if not self.labels and class_col:
            self.labels = len(set(df[class_col]))
        if not self.rows:
            self.rows = len(df)
        if not self.columns:
            self.columns = len(df.columns) if not class_col else len(df.columns) - 1
        if not self.categorical_column_names:
            self.categorical_column_names = ','.join([i for i in df.columns if i != class_col])
        if not self.categorical_columns_number:
            self.categorical_columns_number = len(
                self.categorical_column_names.split(',')) if self.categorical_column_names else None
        if not self.numerical_column_names:
            numerical_column_names = [i for i in df.columns if
                                      i not in self.categorical_column_names] if self.categorical_column_names else len(
                df.columns)
            self.numerical_column_names = ','.join(numerical_column_names) if len(numerical_column_names) > 0 else None

        if self.numerical_columns_number:
            self.numerical_columns_number = len(self.numerical_column_names) if self.numerical_column_names else 0
        if not self.is_synthetic:
            self.is_synthetic = False
        if outlier_mark:
            df[self.outliers_column] = np.where(df[self.outliers_column] == outlier_mark, -1, df[self.outliers_column])
        if not self.outliers:
            self.outliers = len(df[df[self.outliers_column] == -1]) if self.outliers_column else 0

    def prepare_data_and_save(self, df: pd.DataFrame, dest_path: Union[Path, bytes, str],
                              class_col: Optional[str] = None,
                              outlier_mark: Optional[Any] = None) -> None:
        """
        Prepares the dataset for saving by renaming columns and handling outliers,
        then saves the DataFrame to the specified destination.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to process.

        dest_path : Union[Path, bytes, str]
            The destination path where the processed dataset will be saved.

        class_col : Optional[str], optional
            The column name containing the class labels for classification tasks.

        outlier_mark : Optional[Any], optional
            The value used to mark outliers in the dataset.
        """
        if class_col:
            df.rename(columns={class_col: 'class'}, inplace=True)
        elif self.outliers_column:
            df.rename(columns={self.outliers_column: 'outlier'}, inplace=True)
            if outlier_mark:
                df[self.outliers_column] = np.where(df[self.outliers_column] == outlier_mark, -1, 1)

        df.to_csv(dest_path, index=False)