# Data guide

Download and generate data by running the notebooks from the repository root. Only this guide is versioned; datasets and caches remain local.

| Producer | Output under `data/` | Used by |
| --- | --- | --- |
| `dataset_creation.ipynb` | `raw/statcast_2018.parquet` through `raw/statcast_2025.parquet` | The merge step in the same notebook |
| `dataset_creation.ipynb` | `full_dataset.parquet/` | EDA, preprocessing, and clustering |
| `preprocessing_refined.ipynb` | `temporal_full_traintest/train/` and `test/` | Physical pitch classification and strike-zone experiments |
| `preprocessing_refined.ipynb` | `temporal_reduced_traintest/train/` and `test/` | Contextual pitch classification |

The train splits contain 2018–2024; the test splits contain 2025. Numerical imputation and categorical encoding are fitted on the training data.

Spark Parquet outputs are directories containing partition files and completion markers. Preserve the entire directory when moving a dataset. Dataset creation and preprocessing overwrite their respective generated outputs when rerun.

The source is MLB Statcast, downloaded using pybaseball. No data is bundled with this repository; consult the source providers' terms before redistributing downloaded data.
