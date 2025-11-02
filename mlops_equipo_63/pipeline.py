# mlops_equipo_63/pipeline.py
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

from .Configuration import Config
from .load_and_preparation import load_data, prepare_numeric_df, clip_outliers_iqr
from .Split_and_Dummy import prepare_train_test, baseline_classification
from .mlflow_utils import setup_mlflow_experiment
from .Optuna_Study import run_optuna_study
from .Retrain_and_Evaluate import retrain_and_evaluate_best
from .EDA_Plotting import EDAPlotter  # opcional si usas .eda()

class MLOpsPipeline:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        # artefactos
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_numeric: Optional[pd.DataFrame] = None
        self.df_clipped: Optional[pd.DataFrame] = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.threshold: Optional[float] = None
        self.study = None
        self.best_summary: Optional[Dict[str, Any]] = None
        self.tracking_uri: Optional[str] = None
        self.mlflow_cb = None
        self.final_model = None
        self.final_metrics: Optional[Dict[str, Any]] = None
        self.feature_importance = None

    def load(self) -> "MLOpsPipeline":
        self.df_raw = load_data(self.cfg.data_path)
        return self

    def prepare(self) -> "MLOpsPipeline":
        self.df_numeric, _, _ = prepare_numeric_df(
            self.df_raw, label_col=self.cfg.target_col
        )
        self.df_clipped, _, _ = clip_outliers_iqr(
            self.df_numeric, exclude_cols=(self.cfg.target_col,)
        )
        return self

    def eda(self, show: bool = False) -> "MLOpsPipeline":
        if show and self.df_numeric is not None:
            eda = EDAPlotter(self.df_numeric)
            eda.plot_hist(self.cfg.target_col)
            eda.plot_boxplots()
            eda.correlation(plot=True)
        return self

    def split_and_baseline(self) -> "MLOpsPipeline":
        (self.X_train, self.X_test, self.y_train, self.y_test,
         self.threshold, _) = prepare_train_test(
            self.df_clipped,
            target_col=self.cfg.target_col,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )
        base_metrics, _ = baseline_classification(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        print("Baseline:", base_metrics)
        return self

    def setup_tracking(self) -> "MLOpsPipeline":
        self.tracking_uri, self.mlflow_cb = setup_mlflow_experiment(
            experiment_name=self.cfg.mlflow_experiment,
            tracking_dir=self.cfg.mlflow_tracking_uri,
            metric_name="auc",
        )
        return self

    def optimize(self) -> "MLOpsPipeline":
        self.study, self.best_summary = run_optuna_study(
            self.X_train, self.y_train,
            study_name=self.cfg.study_name,
            n_trials=self.cfg.n_trials,
            cv=self.cfg.cv_folds,
            metric_name="roc_auc",
            extra_metrics=("accuracy",),
            enable_models=("RandomForest", "MLP", "XGBoost", "LightGBM"),
            random_state=self.cfg.random_state,
            mlflow_callback=self.mlflow_cb,
            n_jobs_cv=-1,
        )
        print("Best:", self.best_summary)
        return self

    def retrain_and_evaluate(self) -> "MLOpsPipeline":
        self.final_model, self.final_metrics, self.feature_importance = retrain_and_evaluate_best(
            self.study,
            self.X_train, self.y_train, self.X_test, self.y_test,
            feature_names=list(self.X_train.columns),
            experiment_name=self.cfg.mlflow_experiment,
            tracking_uri=Path(self.cfg.mlflow_tracking_uri).resolve().as_uri(),
            parent_from_best_trial=True,
        )
        print("Final:", self.final_metrics)
        return self

    def run_all(self, show_eda: bool = False) -> "MLOpsPipeline":
        return (self.load()
                    .prepare()
                    .eda(show=show_eda)
                    .split_and_baseline()
                    .setup_tracking()
                    .optimize()
                    .retrain_and_evaluate())