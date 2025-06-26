import os

database = 'data/project_db.db'
datasets_table_name = "datasets"
metrics_table_name = "metrics"
comparison_table_name = "comparison"
labeling_table_name = "labeling_kes2025_1_v2"
clustering_table_name = "clustering"
cut_comparison_table_name = "cut_comparison_kes2025_1_v2"
metric_evaluation_table_name = "metric_evaluation"
metrics_custom_table_name = "metrics_custom_kes2025_1"
outliers_table_name = "outliers_ecai2025"
contrastive_outlier_score_table_name = "contrastive_outlier_score"
data_dir = 'data'
raw_data_dir = os.path.join(data_dir, 'raw')
