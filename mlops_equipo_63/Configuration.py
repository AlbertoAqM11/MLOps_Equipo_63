import os
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "C:\\Users\\betoa\\Documents\\TEC\\MLOPS\\Gits\\Fase2\\MLOps_Equipo_63\\references\\online_news_modified.csv"
# original path: r"C:\Users\betoa\Documents\TEC\MLOPS\Gits\Fase2\MLOps_Equipo_63\references\online_news.csv"
    target_col: str = "shares"
    pos_label_threshold: int = 1400
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 50
    cv_folds: int = 5
    study_name: str = "fase2_optuna_study"
    mlflow_experiment: str = "Equipo63_Fase2"
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")