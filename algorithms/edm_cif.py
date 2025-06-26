import numpy as np
import pandas as pd
import math
from typing import List, Optional, Set, Union


class _ITreeNode:
    """
    A class representing a node in an Iterative Tree (I-Tree) structure.

    Attributes:
    ----------
    depth : int
        The depth of the node in the tree.

    size : int
        The size of the node (could represent the number of data points or items in this node).

    feature : Optional[str]
        The feature associated with this node (used for splitting data). This is `None` for leaf nodes.

    partitions : List[Set[str]], optional
        A list of partitions (sets of features or values) that are used for dividing the data at this node.
        This is an empty list by default.

    children : List[_ITreeNode], optional
        The list of child nodes connected to this node. This is an empty list by default.

    Methods:
    -------
    __init__(self, depth: int = 0, size: int = 0, feature: Optional[str] = None,
             partitions: Optional[List[Set[str]]] = None, children: Optional[List['_ITreeNode']] = None):
        Initializes the node with the given parameters.
    """

    def __init__(self, depth: int = 0, size: int = 0, feature: Optional[str] = None,
                 partitions: Optional[List[Set[str]]] = None, children: Optional[List['_ITreeNode']] = None):
        """
        Initializes a node in the Iterative Tree structure.

        Parameters:
        ----------
        depth : int, optional (default=0)
            The depth of the node in the tree.

        size : int, optional (default=0)
            The size of the node (could represent the number of data points or items).

        feature : Optional[str], optional (default=None)
            The feature associated with this node, used for splitting data. `None` means this is a leaf node.

        partitions : Optional[List[Set[str]]], optional (default=None)
            A list of partitions (sets of features or values) used for dividing the data at this node.

        children : Optional[List[_ITreeNode]], optional (default=None)
            A list of child nodes connected to this node. Defaults to an empty list.

        """
        self.depth = depth
        self.size = size
        self.feature = feature
        self.partitions = partitions or []
        self.children = children or []


class EDM_CIF:
    """
    Extended Isolation Forest for Categorical Data (EDM-CIF)
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_samples: Union[int, str] = 'auto',
                 max_features: Union[int, float] = 1.0,
                 contamination: float = 0.1,
                 random_state: Optional[int] = None,
                 partitioning_split: bool = False,
                 n_candidates: int = 10,
                 h_entropy: bool = True,
                 dhlim: bool = True,
                 multi_split: bool = True,
                 m: int = 2):
        """
        Initializes the EDM_CIF model with the given parameters.

        Parameters:
        ----------
        n_estimators : int, optional (default=100)
            The number of trees in the forest.

        max_samples : Union[int, str], optional (default='auto')
            The number of samples to draw from the dataset for each tree.

        max_features : Union[int, float], optional (default=1.0)
            The number of features to consider when looking for the best split.

        contamination : float, optional (default=0.1)
            The proportion of outliers in the dataset.

        random_state : Optional[int], optional (default=None)
            The seed for random number generation.

        partitioning_split : bool, optional (default=False)
            Whether to use partitioning splits during tree construction.

        n_candidates : int, optional (default=10)
            The number of candidate features to consider when partitioning the data.

        h_entropy : bool, optional (default=True)
            Whether to use entropy-based splitting criteria.

        dhlim : bool, optional (default=True)
            Whether to use dynamic high limit for splitting.

        multi_split : bool, optional (default=True)
            Whether to allow multiple splits at each node.

        m : int, optional (default=2)
            The minimum size for a split.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.partitioning_split = partitioning_split
        self.n_candidates = n_candidates
        self.h_entropy = h_entropy
        self.dhlim = dhlim
        self.multi_split = multi_split
        self.m = max(2, m)
        self.random_state = np.random.RandomState(random_state)
        self.trees_: List[_ITreeNode] = []
        self.threshold_: Optional[float] = None
        self.feature_weights_: Optional[np.ndarray] = None

        self.max_samples_ = None
        self.max_features_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Fit the Isolation Forest on the input dataset.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input dataset with categorical features.

        Returns
        -------
        self : object
            Fitted model instance.
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        n_samples, n_features = X.shape

        self.max_samples_ = min(256, n_samples) if self.max_samples == 'auto' else min(self.max_samples, n_samples)
        self.max_features_ = max(1, int(self.max_features * n_features)) if isinstance(self.max_features, float) \
            else min(self.max_features, n_features)

        if self.h_entropy:
            entropies = -X.apply(lambda col: (col.value_counts(normalize=True) *
                                              np.log(col.value_counts(normalize=True))).sum()).fillna(0)
            self.feature_weights_ = entropies / entropies.sum() if entropies.sum() > 0 else np.ones(n_features) / n_features

        self.trees_ = []
        for _ in range(self.n_estimators):
            X_sample = X.sample(n=self.max_samples_, replace=False, random_state=self.random_state)
            base_hlim = math.ceil(math.log2(self.max_samples_))
            hlim = self.random_state.randint(int(0.5 * base_hlim), int(1.5 * base_hlim) + 1) if self.dhlim else base_hlim

            if self.h_entropy:
                feat_idx = self.random_state.choice(n_features, self.max_features_, replace=False, p=self.feature_weights_)
            else:
                feat_idx = self.random_state.choice(n_features, self.max_features_, replace=False)
            features = X.columns[feat_idx]

            tree = self._build_tree(X_sample, features, depth=0, hlim=hlim)
            self.trees_.append(tree)

        scores = self.decision_function(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def _build_tree(self, X: pd.DataFrame, features: List[str], depth: int, hlim: int) -> _ITreeNode:
        """
        Recursively builds a single isolation tree.

        Parameters
        ----------
        X : pd.DataFrame
            Subset of data for the current node.
        features : List[str]
            Features eligible for splitting.
        depth : int
            Current depth of the tree.
        hlim : int
            Maximum depth limit.

        Returns
        -------
        _ITreeNode
            Root of the subtree.
        """
        if depth >= hlim or X.shape[0] <= 1 or len(features) == 0:
            return _ITreeNode(depth=depth, size=X.shape[0])

        feat = self.random_state.choice(features)
        uniques = X[feat].unique()
        if len(uniques) <= 1:
            return _ITreeNode(depth=depth, size=X.shape[0])

        if self.partitioning_split:
            best_parts, best_score = None, -np.inf
            for _ in range(self.n_candidates):
                assign = self.random_state.randint(0, self.m, size=len(uniques))
                parts = [set(uniques[assign == i]) for i in range(self.m) if np.any(assign == i)]
                if len(parts) < 2:
                    continue
                sizes = [X[feat].isin(p).sum() for p in parts]
                score = np.var(sizes)
                if score > best_score:
                    best_score, best_parts = score, parts
            if best_parts is None:
                uniques = self.random_state.permutation(uniques)
                cut = self.random_state.randint(1, len(uniques))
                best_parts = [set(uniques[:cut]), set(uniques[cut:])]
        else:
            uniques = self.random_state.permutation(uniques)
            if self.multi_split:
                best_parts = [set() for _ in range(self.m)]
                for i, val in enumerate(uniques):
                    best_parts[i % self.m].add(val)
            else:
                cut = self.random_state.randint(1, len(uniques))
                best_parts = [set(uniques[:cut]), set(uniques[cut:])]

        node = _ITreeNode(depth=depth, size=X.shape[0], feature=feat, partitions=best_parts)
        for part in best_parts:
            subX = X[X[feat].isin(part)]
            node.children.append(self._build_tree(subX, features, depth + 1, hlim))
        return node

    def _path_length(self, x: pd.Series, node: _ITreeNode) -> float:
        """
        Compute the path length of a sample in a single tree.

        Parameters
        ----------
        x : pd.Series
            A single observation.
        node : _ITreeNode
            Root of the tree.

        Returns
        -------
        float
            Path length.
        """
        if not node.children:
            return node.depth + self._c(node.size)
        for part, child in zip(node.partitions, node.children):
            if x[node.feature] in part:
                return self._path_length(x, child)
        smallest = min(node.children, key=lambda c: c.size)
        return self._path_length(x, smallest)

    def _c(self, n: int) -> float:
        """
        Average path length of unsuccessful search in a binary search tree.

        Parameters
        ----------
        n : int
            Number of samples in a node.

        Returns
        -------
        float
            Correction factor.
        """
        if n <= 1:
            return 0.0
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute anomaly scores for each sample.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        X = pd.DataFrame(X)
        return np.array([
            2 ** (-np.mean([self._path_length(row, tree) for tree in self.trees_]) / self._c(self.max_samples_))
            for _, row in X.iterrows()
        ])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict whether a sample is an outlier or not.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            -1 for outliers, 1 for inliers.
        """
        scores = self.decision_function(X)
        return np.where(scores >= self.threshold_, -1, 1)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return normalized anomaly scores between 0 and 1.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Normalized anomaly scores.
        """
        scores = self.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit the model and predict outliers.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Predictions: -1 for outliers, 1 for inliers.
        """
        self.fit(X)
        return self.predict(X)
