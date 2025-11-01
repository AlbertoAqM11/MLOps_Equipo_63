# Orchestrator.py (raíz del repo)
from pathlib import Path
from mlops_equipo_63 import (
    Config, load_data, prepare_numeric_df, clip_outliers_iqr,
    prepare_train_test, baseline_classification,
    setup_mlflow_experiment, run_optuna_study, retrain_and_evaluate_best,
    EDAVisualizer,
    print_non_numeric, print_missing_info, print_outlier_summary,
    print_baseline, print_best_summary, print_final_metrics,
)

def main():
    # 0) Config
    cfg = Config()
    print(cfg)

    # 1) Cargar datos
    df_raw = load_data(cfg.data_path)

    # 2) Preparación numérica (sin prints dentro de la función)
    df_numeric, non_num, missing_pct = prepare_numeric_df(df_raw, label_col=cfg.target_col)
    print_non_numeric(non_num)
    print_missing_info(missing_pct)

    # 3) EDA opcional (visual)
    # visual = EDAVisualizer(df_numeric)
    # visual.hist(cfg.target_col)
    # visual.boxplots()
    # visual.corr_matrix()

    # 4) Clipping de outliers (IQR)
    df_clip, out_stats, mean_out = clip_outliers_iqr(
        df_numeric, exclude_cols=(cfg.target_col,)
    )
    print_outlier_summary(mean_out)

    # 5) Split + baseline
    X_train, X_test, y_train, y_test, threshold, df_ready = prepare_train_test(
        df_clip, target_col=cfg.target_col, test_size=cfg.test_size, random_state=cfg.random_state
    )
    base_metrics, _ = baseline_classification(X_train, y_train, X_test, y_test)
    print_baseline(base_metrics)

    # 6) MLflow + Optuna
    tracking_uri, mlflow_cb = setup_mlflow_experiment(
        experiment_name=cfg.mlflow_experiment,
        tracking_dir=cfg.mlflow_tracking_uri,
        metric_name="auc",
    )
    study, best = run_optuna_study(
        X_train, y_train,
        study_name=cfg.study_name,
        n_trials=cfg.n_trials,
        cv=cfg.cv_folds,
        metric_name="roc_auc",
        extra_metrics=("accuracy",),
        enable_models=("RandomForest", "MLP", "XGBoost", "LightGBM"),
        random_state=cfg.random_state,
        mlflow_callback=mlflow_cb,
        n_jobs_cv=-1,
    )
    print_best_summary(best)

    # 7) Reentrenar mejor modelo + evaluación final
    final_model, final_metrics, feat_imp = retrain_and_evaluate_best(
        study,
        X_train, y_train, X_test, y_test,
        feature_names=list(X_train.columns),
        experiment_name=cfg.mlflow_experiment,
        tracking_uri=Path(cfg.mlflow_tracking_uri).resolve().as_uri(),
        parent_from_best_trial=True,
    )
    print_final_metrics(final_metrics)
    if feat_imp is not None:
        print("\nTop importances:\n", feat_imp.head(10))

if __name__ == "__main__":
    main()