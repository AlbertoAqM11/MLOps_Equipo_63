# mlops_equipo_63/reporting.py
from typing import Dict, Any, Iterable
import pandas as pd

def print_non_numeric(non_numeric_cols: Iterable):
    cols = list(non_numeric_cols) if non_numeric_cols is not None else []
    print("Columnas no numéricas (muestras):", cols[:10])

def print_missing_info(missing_pct: pd.Series):
    print("\nPorcentaje de valores faltantes por columna (>0):")
    if missing_pct is None or missing_pct.empty:
        print("(sin valores faltantes)")
        return
    mp = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if mp.empty:
        print("(sin valores faltantes)")
    else:
        print(mp)

def print_outlier_summary(mean_pct: float):
    try:
        print(f"Promedio de outliers recortados: {float(mean_pct):.2f}%")
    except Exception:
        print("Promedio de outliers recortados: (valor no numérico)")

def print_baseline(metrics: Dict[str, Any]):
    print("\n=== Rendimiento base ===")
    if not isinstance(metrics, dict):
        print(metrics)
        return
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}:\n{v}")

def print_best_summary(best: Dict[str, Any]):
    print("\n=== Mejor resultado (Optuna) ===")
    if not isinstance(best, dict):
        print(best)
        return
    for k, v in best.items():
        print(f"{k}: {v}")

def print_final_metrics(final_metrics: Dict[str, Any]):
    print("\n=== Evaluación final ===")
    if not isinstance(final_metrics, dict):
        print(final_metrics)
        return
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}:\n{v}")