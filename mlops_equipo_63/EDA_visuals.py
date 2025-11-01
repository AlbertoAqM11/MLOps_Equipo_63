# mlops_equipo_63/eda_visuals.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Iterable, Optional

class EDAVisualizer:
    """Centraliza gráficos exploratorios y métricas visuales."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def hist(self, col: str, bins: int = 100, figsize=(10,6)):
        plt.figure(figsize=figsize)
        plt.hist(self.df[col].dropna(), bins=bins)
        plt.title(f"Distribución de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()

    def boxplots(self, cols: Optional[Iterable[str]] = None, figsize=(6,4)):
        cols = cols or self.df.columns
        for c in cols:
            plt.figure(figsize=figsize)
            sns.boxplot(x=self.df[c].dropna())
            plt.title(f"Boxplot: {c}")
            plt.tight_layout()
            plt.show()

    def corr_matrix(self, method="pearson", figsize=(20,20)):
        corr = self.df.corr(numeric_only=True, method=method)
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title(f"Matriz de correlación ({method})")
        plt.tight_layout()
        plt.show()
        return corr
    