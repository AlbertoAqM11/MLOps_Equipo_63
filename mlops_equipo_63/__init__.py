# mlops_equipo_63/__init__.py
from .Configuration import Config
from .load_and_preparation import (
    load_data,
    prepare_numeric_df,   # ahora devuelve (df_numeric, non_numeric_cols, missing_pct)
    clip_outliers_iqr,
)
from .Split_and_Dummy import prepare_train_test, baseline_classification
from .mlflow_utils import setup_mlflow_experiment
from .Optuna_Study import run_optuna_study
from .Retrain_and_Evaluate import retrain_and_evaluate_best

# módulos de presentación (nuevos)
from .EDA_visuals import EDAVisualizer
from .reporting import (
    print_non_numeric,
    print_missing_info,
    print_outlier_summary,
    print_baseline,
    print_best_summary,
    print_final_metrics,
)

__all__ = [
    # core config
    "Config",
    # preparación de datos
    "load_data", "prepare_numeric_df", "clip_outliers_iqr",
    # split + baseline
    "prepare_train_test", "baseline_classification",
    # tracking/optimización
    "setup_mlflow_experiment", "run_optuna_study", "retrain_and_evaluate_best",
    # presentación/EDA
    "EDAVisualizer",
    "print_non_numeric", "print_missing_info", "print_outlier_summary",
    "print_baseline", "print_best_summary", "print_final_metrics",
]