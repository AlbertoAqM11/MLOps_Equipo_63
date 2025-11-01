from typing import Iterable, Tuple, Dict, Any
import numpy as np
import optuna

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Modelos opcionales (si no están instalados, los tratamos como no disponibles)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

def run_optuna_study(
    X_train, y_train,
    study_name: str = "optuna_study",
    n_trials: int = 50,
    cv: int = 3,
    metric_name: str = "roc_auc",                 # métrica objetivo de Optuna
    extra_metrics: Iterable[str] = ("accuracy",), # métricas adicionales
    enable_models: Iterable[str] = ("RandomForest", "MLP", "XGBoost", "LightGBM"),
    random_state: int = 42,
    mlflow_callback=None,                         # puede ser el callback de MLflow o None
    n_jobs_cv: int = -1
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Lanza una búsqueda con Optuna y devuelve (study, best_summary).
    Registra en MLflow si pasas un callback en 'mlflow_callback'.
    """

    # mapa seguro para MLP (evita eval)
    HIDDEN_MAP = {
        "50": (50,),
        "100": (100,),
        "50x50": (50, 50),
    }

    # filtra modelos disponibles (por si no están instalados)
    enabled = []
    for m in enable_models:
        if m == "XGBoost" and xgb is None:
            print("[warn] XGBoost no disponible. Se omite.")
            continue
        if m == "LightGBM" and lgb is None:
            print("[warn] LightGBM no disponible. Se omite.")
            continue
        enabled.append(m)
    if not enabled:
        raise ValueError("No hay modelos habilitados/instalados para probar.")

    scoring = [metric_name, *extra_metrics]

    def build_estimator(trial) -> Pipeline:
        """Crea el estimador (opcionalmente con scaler) según el modelo elegido."""
        name = trial.suggest_categorical("classifier", enabled)

        if name == "RandomForest":
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int("rf_n_estimators", 50, 400),
                max_depth=trial.suggest_int("rf_max_depth", 5, 30),
                random_state=random_state,
                n_jobs=-1,
            )
            steps = [("classifier", clf)]  # RF no necesita scaler
        elif name == "MLP":
            hidden_key = trial.suggest_categorical("mlp_hidden_layers", list(HIDDEN_MAP.keys()))
            clf = MLPClassifier(
                hidden_layer_sizes=HIDDEN_MAP[hidden_key],
                alpha=trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True),
                max_iter=300,
                early_stopping=True,
                random_state=random_state,
            )
            steps = [("scaler", StandardScaler()), ("classifier", clf)]
        elif name == "XGBoost":
            clf = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("xgb_n_estimators", 100, 800),
                learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
                max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
                subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )
            steps = [("classifier", clf)]  # XGB no necesita scaler
        elif name == "LightGBM":
            clf = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("lgbm_n_estimators", 100, 800),
                learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.3),
                num_leaves=trial.suggest_int("lgbm_num_leaves", 20, 200),
                subsample=trial.suggest_float("lgbm_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("lgbm_colsample", 0.6, 1.0),
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
            )
            steps = [("classifier", clf)]
        else:
            # fallback para que nunca falle
            clf = DummyClassifier(strategy="stratified", random_state=random_state)
            steps = [("classifier", clf)]

        return Pipeline(steps=steps)

    def objective(trial):
        # si usas MLflow callback no hace falta start_run aquí; el callback se encarga
        pipeline = build_estimator(trial)

        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=n_jobs_cv, error_score="raise"
        )

        mean_target = float(np.mean(scores[f"test_{metric_name}"]))
        # guarda métricas extra en user_attrs para imprimir luego
        for m in extra_metrics:
            trial.set_user_attr(m, float(np.mean(scores[f"test_{m}"])))
        return mean_target

    study = optuna.create_study(direction="maximize", study_name=study_name)
    callbacks = []
    if mlflow_callback is not None:
        callbacks.append(mlflow_callback)
    # callback para ver progreso por consola
    def print_metrics_callback(study, trial):
        extras = " ".join(
            f"{m}: {trial.user_attrs.get(m, float('nan')):.4f}" for m in extra_metrics
        )
        print(f"Trial {trial.number} → {metric_name.upper()}: {trial.value:.4f} | {extras}")

    callbacks.append(print_metrics_callback)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=callbacks)

    # resumen “bonito”
    best = study.best_trial
    summary = {
        "metric": metric_name,
        "best_value": float(best.value),
        "best_params": best.params,
        **{f"cv_{m}": float(best.user_attrs.get(m)) for m in extra_metrics},
        "n_trials": len(study.trials),
    }
    return study, summary