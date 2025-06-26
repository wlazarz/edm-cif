from consts.const_variables import *

create_datasets_table_query = f'''
CREATE TABLE {datasets_table_name} (
    dataset_id TEXT PRIMARY KEY,
    dataset_name TEXT,
    extension TEXT,
    path TEXT,
    labels INTEGER,
    rows INTEGER,
    columns INTEGER,
    numerical_column_names TEXT,
    numerical_columns_number INTEGER,
    categorical_column_names TEXT,
    categorical_columns_number INTEGER,
    is_synthetic INTEGER,
    outliers_column TEXT,
    outliers REAL,
    http_path TEXT,
    currdate TEXT);
'''

create_outliers_table_query = f'''
CREATE TABLE {outliers_table_name} (
    process_id TEXT,
    algorithm TEXT,
    task TEXT,
    split_mode TEXT,
    split_method TEXT,
    dataset TEXT,
    dataset_type TEXT,
    n_estimators INT,
    max_samples INT,
    min_cluster_size INT,
    contamination REAL,
    max_features INT,
    n_neighbors INT,
    alpha REAL,
    beta REAL,
    epsilon REAL,
    theta REAL,
    k INT,
    m INT,
    metric TEXT,
    outliers_threshold INT,
    ground_truth INT,
    clustering INT,
    time_sec REAL,  
    detected_outliers_num INT,
    detected_outliers_perc REAL,
    real_outliers_num INT,
    real_outliers_perc REAL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    tp INT,
    tn INT,
    fp INT,
    fn INT,
    tpr REAL,
    tnr REAL,
    fpr REAL,
    fnr REAL,
    clusters INT,
    silhouette_score REAL,
    calinski_harabasz_score REAL,
    davies_bouldin_score REAL,
    dunn_index REAL,
    cluster_entropy REAL,
    fmi REAL,
    ari REAL,
    vi REAL,
    nmi REAL,
    shannon REAL,
    distribution TEXT);
'''

create_contrastive_outlier_score_query = f'''
CREATE TABLE {contrastive_outlier_score_table_name} (
    process_id TEXT,
    metric TEXT,
    strategy TEXT,
    n INT,
    contrastive_outlier_score REAL);
'''


query_table_exists = '''
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' AND name='{0}';'''

query_insert_record = '''INSERT INTO {0} ({1}) VALUES ({2})'''

