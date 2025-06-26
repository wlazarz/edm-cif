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

create_labeling_table_query = f'''
CREATE TABLE {labeling_table_name} (
    process_id TEXT,
    method TEXT,
    algorithm TEXT,
    dataset TEXT,
    input_samples REAL,
    iters INTEGER,
    strategy TEXT,
    metric TEXT,
    input_samples_method TEXT,
    task TEXT, 
    params TEXT);
'''

create_class_evaluation_table_query = f'''
CREATE TABLE {metric_evaluation_table_name} (
    process_id TEXT,
    non_error_ratio REAL,
    algorithm TEXT,
    alg_param_1 REAL,
    alg_param_2 REAL,
    dataset TEXT,
    task TEXT, 
    params TEXT);
'''

create_metrics_table_query = f'''
CREATE TABLE {metrics_table_name} (
    process_id TEXT,
    silhouette_score REAL, 
    silhouette_score_hamming REAL, 
    calinski_harabasz_score REAL, 
    calinski_harabasz_score_hamming REAL, 
    davies_bouldin_score REAL, 
    davies_bouldin_score_hamming REAL, 
    dunn_index REAL, 
    dunn_index_hamming REAL, 
    cluster_entropy REAL, 
    cluster_inconsistency REAL, 
    cluster_separation REAL, 
    profit REAL, 
    hamming_ratio REAL, 
    m1 REAL, 
    m2 REAL, 
    m3 REAL, 
    m4 REAL);
'''


metrics_custom_table_query = f'''
CREATE TABLE {metrics_custom_table_name} (
    process_id TEXT,
    metric TEXT,
    silhouette_score REAL, 
    calinski_harabasz_score REAL, 
    davies_bouldin_score REAL, 
    dunn_index REAL);
'''


create_comparison_table_query = f'''
CREATE TABLE {comparison_table_name} (
    process_id TEXT,
    task TEXT,
    clusters1 INT,
    clusters2 INT,
    accuracy REAL,
    macro_precision REAL,
    micro_precision REAL,
    macro_recall REAL,
    micro_recall REAL,
    macro_f1 REAL,
    micro_f1 REAL,
    auc_roc REAL,
    fmi REAL,
    ari REAL,
    vi REAL,
    nmi REAL,
    shannon REAL,
    cohens_kappa REAL,
    jaccard REAL,
    matthews_corrcoef REAL,
    distribution TEXT);
'''

create_cut_comparison_table_query = f'''
CREATE TABLE {cut_comparison_table_name} (
    process_id TEXT,
    cut_point REAL,
    n_elements INT,
    task TEXT,
    clusters1 INT,
    clusters2 INT,
    accuracy REAL,
    macro_precision REAL,
    micro_precision REAL,
    macro_recall REAL,
    micro_recall REAL,
    macro_f1 REAL,
    micro_f1 REAL,
    auc_roc REAL,
    fmi REAL,
    ari REAL,
    vi REAL,
    nmi REAL,
    shannon REAL,
    cohens_kappa REAL,
    jaccard REAL,
    matthews_corrcoef REAL,
    distribution TEXT);
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

