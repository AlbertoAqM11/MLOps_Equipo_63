from pathlib import Path
import mlflow
from optuna.integration.mlflow import MLflowCallback

def setup_mlflow_experiment(
    experiment_name: str = "Online_News_Popularity_Estudio_Optuna",
    tracking_dir: str = "mlruns",
    metric_name: str = "auc",
):
    tracking_abs = Path(tracking_dir).resolve()

    # URI correcta en Windows/Mac/Linux
    tracking_uri = tracking_abs.as_uri()  # p.ej. 'file:///C:/Users/.../mlruns'

    mlflow.set_tracking_uri(tracking_uri)

    # Desactiva el Model Registry (no soportado en file store)
    mlflow.set_registry_uri("")  # esquema vacío = sin registry

    mlflow.set_experiment(experiment_name)

    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=metric_name
    )
    return tracking_uri, mlflow_cb