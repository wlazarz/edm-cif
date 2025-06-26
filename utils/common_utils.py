import pandas as pd
import numpy as np
from typing import Tuple,Dict, List


def variable_coding(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], str]]:
    """
    Converts categorical variables in the DataFrame to numeric codes.

    Each unique value in a categorical column is assigned a unique integer value. The mapping between
    the original value and the encoded integer is stored in a dictionary.

    Parameters:
    ----------
    X : pd.DataFrame
        A pandas DataFrame containing categorical variables to be encoded.

    Returns:
    -------
    Tuple[pd.DataFrame, Dict[Tuple[str, str], str]]
        - The DataFrame with categorical variables replaced by numeric codes.
        - A dictionary mapping (column name, encoded value) pairs to the original categorical values.
    """
    val_dict = {}
    i = 1
    for col in X.columns:
        uniq_val = X[col].value_counts().keys().tolist()
        for val in uniq_val:
            X.loc[X[col] == val, col] = str(i)
            val_dict[(col, str(i))] = val
            i += 1
    return X, val_dict


def ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies One-Hot Encoding (OHE) to the DataFrame by encoding categorical variables as one-hot vectors.

    If a column contains missing values, they are replaced with the mode (most frequent value) of that column.

    Parameters:
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing categorical data to be encoded.

    Returns:
    -------
    pd.DataFrame
        The DataFrame after applying One-Hot Encoding to categorical variables.
    """
    ohe_data = df.copy()
    for c in ohe_data.columns:
        col_mode = df[c].mode().values[0]
        df[c].fillna(col_mode, inplace=True)
        color_encoded = pd.get_dummies(ohe_data[c], prefix=c)
        ohe_data = pd.concat([ohe_data, color_encoded], axis=1).drop(columns=[c])

    return ohe_data


def read_datasets_to_dict(db_session) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.Series]],
Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Reads datasets from the database, processes them into pandas DataFrames, and applies One-Hot Encoding (OHE).

    The datasets are categorized into 'synthetic' and 'real', with the OHE applied to each.
    Additionally, the original datasets and class labels are stored in dictionaries.

    Parameters:
    ----------
    db_session : object
        The database session object used to query and fetch data from the database.

    Returns:
    -------
    Tuple[Dict[str, Dict[str, pd.DataFrame]],
          Dict[str, Dict[str, pd.Series]],
          Dict[str, Dict[str, pd.DataFrame]]]
        - A tuple containing three dictionaries:
          1. `datasets`: A dictionary of original datasets (after variable coding).
          2. `datasets_class`: A dictionary of class labels corresponding to each dataset.
          3. `ohe_datasets`: A dictionary of datasets after applying One-Hot Encoding.
    """
    datasets = {'synthetic': {}, 'real': {}}
    ohe_datasets = {'synthetic': {}, 'real': {}}
    datasets_class = {'synthetic': {}, 'real': {}}
    query = """SELECT * FROM datasets"""

    for row in db_session.select_from_table(query, how='many'):
        df = pd.read_csv(row['path'], sep=';')
        name = row['dataset_name']
        class_ = df['class']
        is_synthetic = 'synthetic' if row['is_synthetic'] else 'real'
        df.drop('class', axis=1, inplace=True)

        df = (df.replace('?', None)
              .replace('undefined', None)
              .replace('unknown', None)
              .replace('nan', None)
              .replace(np.nan, None))

        ohe_data = ohe(df)
        df, _ = variable_coding(df)

        ohe_datasets[is_synthetic][name] = ohe_data
        datasets[is_synthetic][name] = df.astype(str)
        datasets_class[is_synthetic][name] = class_.astype(str)

    return datasets, datasets_class, ohe_datasets


def ohe_of_subset_of_df(original_columns: List[str], ohe_columns: List[str], data: np.ndarray) -> np.ndarray:
    """
    Computes the One-Hot Encoding (OHE) for a subset of columns in a DataFrame and returns the encoded subset as a NumPy array.

    This function first converts the input data into a DataFrame with the specified original columns,
    then applies One-Hot Encoding (OHE) to the DataFrame, and reindexes the result to match the specified OHE columns.

    Parameters:
    ----------
    original_columns : List[str]
        A list of column names that correspond to the original columns in the data.

    ohe_columns : List[str]
        A list of column names that represent the desired columns in the output after OHE.

    data : np.ndarray
        A 2D numpy array containing the data to be One-Hot Encoded.

    Returns:
    -------
    np.ndarray
        A 2D numpy array containing the One-Hot Encoded subset of the input data, with columns matching `ohe_columns`.
    """

    data = pd.DataFrame(data, columns=original_columns)
    encoded_subset = pd.get_dummies(data, dtype=int)
    encoded_subset = encoded_subset.reindex(columns=ohe_columns, fill_value=0)
    encoded_subset_np = encoded_subset.to_numpy()

    return encoded_subset_np