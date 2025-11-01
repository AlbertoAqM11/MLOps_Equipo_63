import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def prepare_train_test(df: pd.DataFrame, 
                       target_col: str = 'shares', 
                       test_size: float = 0.2, 
                       random_state: int = 42, 
                       stratify: bool = True, 
                       verbose: bool = True):
    "Prepares training and testing datasets for binary classification."
    df_clean = df.copy()

    # --- Step 1: Handle Missing Values in the Target ---
    if verbose:
        print(f"Filas antes de limpiar: {len(df_clean)}")
        print(f"Número de filas con '{target_col}' faltantes: {df_clean[target_col].isnull().sum()}")

    df_clean.dropna(subset=[target_col], inplace=True)

    if verbose:
        print(f"Filas después de limpiar: {len(df_clean)}")

    # --- Step 2: Create Binary Classification Target ---
    threshold = df_clean[target_col].median()
    df_clean['popular'] = (df_clean[target_col] > threshold).astype(int)

    if verbose:
        print(f"\nUsando umbral de popularidad = {threshold:.0f} ({target_col} mediana).")
        print("\nDistribución de clases binarias:")
        print(df_clean['popular'].value_counts(normalize=True).rename('proporción'))

    # --- Step 3: Define X and y ---
    X = df_clean.drop([target_col, 'popular'], axis=1)
    y = df_clean['popular']

    # --- Step 4: Train-Test Split ---
    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    if verbose:
        print(f"\nTamaño del conjunto de entrenamiento: {len(X_train)}")
        print(f"Tamaño del conjunto de prueba: {len(X_test)}")
        print("="*60)

    return X_train, X_test, y_train, y_test, threshold, df_clean

def baseline_classification(
    X_train, y_train, X_test, y_test,
    impute_strategy: str = "mean",
    baseline_strategy: str = "stratified",  # "most_frequent", "uniform", etc.
    random_state: int = 42,
    return_pipeline: bool = False
) -> Tuple[Dict[str, Any], Optional[Pipeline]]:
    """
    Entrena un baseline (DummyClassifier) en un pipeline con imputación
    y devuelve métricas + (opcional) el pipeline entrenado.
    """
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("classifier", DummyClassifier(strategy=baseline_strategy, random_state=random_state))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }

    # Si hay predict_proba y el problema es binario, calculamos AUC
    try:
        proba = pipe.predict_proba(X_test)
        if proba is not None and proba.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
    except Exception:
        pass

    if return_pipeline:
        return metrics, pipe
    return metrics, None