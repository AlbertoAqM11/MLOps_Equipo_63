# mlops_equipo_63/Retrain_and_Evaluate.py
import os
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models.signature import infer_signature
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Modelos opcionales
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def retrain_and_evaluate_best(
    study,
    X_train, y_train, X_test, y_test,
    feature_names: Optional[list] = None,
    experiment_name: str = "Online_News_Popularity_Estudio_Optuna",
    tracking_uri: Optional[str] = None,   # ej: 'file:///C:/.../mlruns'
    parent_from_best_trial: bool = True,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Reconstruye el mejor modelo del estudio, entrena en el train, evalúa en el test,
    guarda gráficos en reports/ (para DVC) y registra artefactos en MLflow.
    Devuelve: (pipeline_final, métricas_dict, importance_df | None)
    """
    # ---------- reconstruir el mejor estimador ----------
    best_params = study.best_params.copy()
    clf_name = best_params.pop("classifier")

    def _parse_mlp_hidden(val):
        # segura: acepta tuplas, '50', '(50,)', '50,50', etc.
        if isinstance(val, tuple):
            return val
        if isinstance(val, str):
            s = val.strip().strip("()")
            if not s:
                return (50,)
            try:
                return tuple(int(x) for x in s.split(",") if x.strip())
            except Exception:
                return (50,)
        return (50,)

    model_params: Dict[str, Any] = {}
    steps = []

    if clf_name == "RandomForest":
        # parámetros venían con prefijo rf_
        rename_map = {
            "rf_n_estimators": "n_estimators",
            "rf_max_depth": "max_depth",
        }
        model_params = {rename_map[k]: v for k, v in best_params.items() if k in rename_map}
        final_clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **model_params)
        # RF no requiere scaler

    elif clf_name == "MLP":
        if "mlp_hidden_layers" in best_params:
            best_params["hidden_layer_sizes"] = _parse_mlp_hidden(best_params.pop("mlp_hidden_layers"))
        # mapear prefijos mlp_
        allow = {"mlp_alpha": "alpha"}
        parsed = {allow[k]: v for k, v in best_params.items() if k in allow}
        # añade otros si existieran en el estudio
        model_params = parsed
        final_clf = MLPClassifier(max_iter=300, early_stopping=True, random_state=random_state, **model_params)
        steps.append(("scaler", StandardScaler()))  # MLP sí requiere escalado

    elif clf_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("XGBoost no está instalado, no puedo reconstruir el mejor modelo.")
        # map explícito para evitar 'colsample' inválido
        rename_map = {
            "xgb_n_estimators": "n_estimators",
            "xgb_learning_rate": "learning_rate",
            "xgb_max_depth": "max_depth",
            "xgb_subsample": "subsample",
            "xgb_colsample": "colsample_bytree",   # evitar warning: 'colsample' no existe
        }
        model_params = {rename_map[k]: v for k, v in best_params.items() if k in rename_map}
        final_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            **model_params
        )
        # XGBoost no requiere scaler

    elif clf_name == "LightGBM":
        if lgb is None:
            raise RuntimeError("LightGBM no está instalado, no puedo reconstruir el mejor modelo.")
        # LightGBM usa aliases comunes: subsample -> bagging_fraction; colsample -> feature_fraction
        rename_map = {
            "lgbm_n_estimators": "n_estimators",
            "lgbm_learning_rate": "learning_rate",
            "lgbm_num_leaves": "num_leaves",
            "lgbm_subsample": "subsample",          # alias válido (bagging_fraction)
            "lgbm_colsample": "feature_fraction",   # equivalente a colsample
        }
        model_params = {rename_map[k]: v for k, v in best_params.items() if k in rename_map}
        final_clf = lgb.LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            **model_params
        )
        # LightGBM no requiere scaler

    else:
        raise ValueError(f"Clasificador desconocido en best_params: {clf_name}")

    steps.append(("classifier", final_clf))
    final_pipeline = Pipeline(steps=steps)

    # ---------- MLflow ----------
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        # si usas file store local, desactiva registry:
        if tracking_uri.startswith("file:") or tracking_uri.startswith("file:///"):
            mlflow.set_registry_uri("")

    mlflow.set_experiment(experiment_name)
    artifact_root = os.path.abspath("artifacts")
    
    parent_run_id = None
    if parent_from_best_trial:
        parent_run_id = study.best_trial.system_attrs.get("mlflow_run_id")

    metrics: Dict[str, Any] = {}
    importance_df: Optional[pd.DataFrame] = None

    with mlflow.start_run(
        run_name=f"Final_{clf_name}_Model",
        nested=True if parent_run_id else False,
        tags={"stage": "final_evaluation"},
        parent_run_id=parent_run_id
    ):
        # parámetros del mejor modelo
        mlflow.log_param("classifier", clf_name)
        for p_k, p_v in model_params.items():
            mlflow.log_param(p_k, p_v)

        # entrenar y evaluar
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)

        # si el modelo soporta proba, calcula AUC
        if hasattr(final_pipeline, "predict_proba"):
            y_proba = final_pipeline.predict_proba(X_test)[:, 1]
            metrics["final_auc"] = float(roc_auc_score(y_test, y_proba))
        else:
            metrics["final_auc"] = float("nan")

        metrics["final_accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["classification_report"] = classification_report(
            y_test, y_pred, target_names=["Unpopular", "Popular"], zero_division=0
        )

        print(f"\nFinal Test AUC: {metrics['final_auc']:.4f}")
        print(f"Final Test Accuracy: {metrics['final_accuracy']:.4f}")
        print("\n--- Classification Report ---")
        print(metrics["classification_report"])

        # log métricas numéricas
        mlflow.log_metric("final_auc", metrics["final_auc"])
        mlflow.log_metric("final_accuracy", metrics["final_accuracy"])

        # --- asegurar carpeta reports ---
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # =========================
        # Matriz de confusión (guardar SIEMPRE en reports/ y log a MLflow)
        # =========================
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Unpopular", "Popular"],
            yticklabels=["Unpopular", "Popular"]
        )
        plt.title("Confusion Matrix on Test Set")
        plt.xlabel("Predicted"); plt.ylabel("True")

        cm_local = reports_dir / "confusion_matrix.png"
        plt.savefig(cm_local, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(cm_local))

        # =========================
        # Importancias de variables (o placeholder)
        # =========================
        clf = final_pipeline.named_steps["classifier"]
        fi_local = reports_dir / "feature_importance.png"

        if hasattr(clf, "feature_importances_") and feature_names is not None:
            importances = clf.feature_importances_
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values("Importance", ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
            plt.title(f"Top 20 Feature Importances — {clf.__class__.__name__}")
            plt.tight_layout()

            # guarda en reports/ (para DVC)
            plt.savefig(fi_local, bbox_inches="tight")
            plt.close()

            # registra en MLflow
            mlflow.log_artifact(str(fi_local))

            # CSV con top-20 (útil para inspección)
            top20_csv = reports_dir / "feature_importance_top20.csv"
            importance_df.head(20).to_csv(top20_csv, index=False)
            mlflow.log_artifact(str(top20_csv))

        else:
            # Si el modelo no tiene importancias, crear placeholder para DVC
            plt.figure(figsize=(8, 5))
            plt.text(
                0.5, 0.5,
                "Feature importances not available\nfor this classifier.",
                ha="center", va="center"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fi_local, bbox_inches="tight")
            plt.close()

            mlflow.log_artifact(str(fi_local))
            importance_df = None

        # --- Artefacto con parámetros completos del best trial (trazabilidad)
        try:
            best_params_path = reports_dir / "best_params.json"
            with open(best_params_path, "w", encoding="utf-8") as f:
                json.dump(study.best_trial.params, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(str(best_params_path))
        except Exception:
            pass

        # --- Log del modelo con firma e input_example (evita warnings)
        try:
            X_example = X_train.head(5)
            y_pred_example = final_pipeline.predict(X_example)
            signature = infer_signature(X_example, y_pred_example)

            # artifact_path sigue siendo válido; algunos MLflow avisan si usas "name".
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="final_model",
                signature=signature,
                input_example=X_example,
            )
        except TypeError:
            # fallback por compatibilidad con versiones viejas de MLflow
            mlflow.sklearn.log_model(final_pipeline, artifact_path="final_model")

    print("\n✅ Modelo final registrado.")
    return final_pipeline, metrics, importance_df