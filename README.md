# EDM-CIF: Extended Isolation Forest for Categorical Data

**EDM-CIF** is an enhanced version of the Isolation Forest algorithm designed specifically for categorical data. It supports entropy-weighted feature selection, dynamic tree height limitation, and multiway splits, making it highly effective for anomaly detection in discrete feature spaces.

## Features

- Handles **purely categorical datasets** without the need for one-hot encoding
- Supports **multi-way splits** for better separation of high-cardinality features
- **Entropy-weighted feature sampling** for informed split decisions
- **Dynamic tree height** for adaptive complexity control
- Optional **partitioning heuristic** for balanced multi-splits
- Scalable, interpretable, and easy to integrate

## Quick Start

```python
from edm_cif import CategoricalIsolationForest
import pandas as pd

# Example categorical dataset
data = pd.DataFrame({
    'Color': ['Red', 'Red', 'Blue', 'Blue', 'Green', 'Green', 'Yellow', 'Pink'],
    'Shape': ['X', 'X', 'Y', 'Y', 'X', 'X', 'Z', 'Z'],
    'Type': ['Cat', 'Dog', 'Dog', 'Cat', 'Cat', 'Dog', 'Fish', 'Fish']
})

model = CategoricalIsolationForest(
    n_estimators=100,
    max_samples='auto',
    max_features=1.0,
    contamination=0.1,
    h_entropy=True,
    dhlim=True,
    multi_split=True,
    m=3
)

predictions = model.fit_predict(data)
scores = model.decision_function(data)
```

## Documentation

**Parameters**

n_estimators: Number of trees in the forest.

max_samples: Number of samples per tree ('auto' or int).

max_features: Fraction or number of features per tree.

contamination: Expected proportion of outliers.

h_entropy: Use entropy-weighted feature sampling.

dhlim: Use dynamic height limits for trees.

multi_split: Use multiway splits (instead of binary).

m: Number of partitions per split (used if multi_split=True).

**Methods**

fit(X): Fit the model.

predict(X): Predict -1 for outliers, 1 for inliers.

decision_function(X): Return anomaly scores.

predict_proba(X): Return normalized scores [0, 1].

fit_predict(X): One-step fit and predict.

