import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Iterable, Dict, Tuple

# mlops_equipo_63/load_and_preparation.py
from typing import Iterable, List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def prepare_numeric_df(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = ("url",),
    label_col: str = "shares",
    drop_cols: Iterable[str] = ("url", "timedelta"),
    impute_strategy: str = "median",
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    "Prepara un DataFrame con solo columnas numéricas, imputando faltantes."
    df_numeric = pd.DataFrame()
    non_numeric_cols: List[str] = []

    # Convertir columnas a numéricas (excepto excluidas)
    for col in df.columns:
        if col in exclude_cols:
            continue
        try:
            df_numeric[col] = pd.to_numeric(df[col], errors="coerce")
        except (ValueError, TypeError):
            non_numeric_cols.append(col)

    # Quitar filas con NaN en el label si existe
    if label_col in df_numeric.columns:
        df_numeric.dropna(subset=[label_col], inplace=True)

    # Eliminar columnas innecesarias si existen
    for c in drop_cols:
        if c in df_numeric.columns:
            df_numeric.drop(columns=[c], inplace=True)

    # Imputar faltantes
    imputer = SimpleImputer(strategy=impute_strategy)
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Calcular % de NaNs post-imputación (debería ser 0, pero lo regresamos por consistencia)
    missing_values_percentage = (df_numeric.isnull().sum() / len(df_numeric)) * 100

    return df_numeric, non_numeric_cols, missing_values_percentage

def clip_outliers_iqr(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = ("shares",),
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    "Recorta outliers usando el método IQR para columnas numéricas."
    dfc = df.copy()
    outlier_percentages: Dict[str, float] = {}

    for col in dfc.select_dtypes(include="number").columns:
        if col in exclude_cols:
            continue
        Q1 = dfc[col].quantile(0.25)
        Q3 = dfc[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask = (dfc[col] < lower) | (dfc[col] > upper)
        outlier_percentages[col] = float(mask.mean() * 100.0)
        dfc[col] = dfc[col].clip(lower=lower, upper=upper)

    mean_pct = float(np.mean(list(outlier_percentages.values()))) if outlier_percentages else 0.0
    return dfc, outlier_percentages, mean_pct