import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Iterable

class EDAPlotter:
    def __init__(self, df: pd.DataFrame, figsize_hist=(10,6), figsize_box=(6,4), corr_method="pearson"):
        self.df = df
        self.figsize_hist = figsize_hist
        self.figsize_box = figsize_box
        self.corr_method = corr_method

    def plot_hist(self, col: str = "shares", bins: int = 100):
        if col not in self.df.columns:
            raise ValueError(f"'{col}' no está en el DataFrame.")
        plt.figure(figsize=self.figsize_hist)
        plt.hist(self.df[col].dropna(), bins=bins)
        plt.title(f"Distribution of {col} (Histogram)")
        plt.xlabel(f"{col} (binned)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, cols: Optional[Iterable[str]] = None):
        cols = list(cols) if cols is not None else list(self.df.columns)
        for c in cols:
            if c not in self.df.columns:
                print(f"[warn] '{c}' no está en el DataFrame, se omite.")
                continue
            plt.figure(figsize=self.figsize_box)
            sns.boxplot(x=self.df[c].dropna())
            plt.title(f"Distribution of {c}")
            plt.xlabel(c)
            plt.ylabel("Value")
            plt.tight_layout()
            plt.show()
            plt.close()

    def correlation(self, plot: bool = False, decimals: int = 2):
        corr = self.df.corr(numeric_only=True, method=self.corr_method)
        if plot:
            plt.rcParams.update({'font.size': 6})
            plt.figure(figsize=(25, 25))
            corr_to_plot = corr.round(decimals) if decimals is not None else corr
            sns.heatmap(corr_to_plot, center=0,annot=True, fmt=".2f")
            plt.title(f"Correlation Matrix ({self.corr_method})")
            plt.tight_layout()
            plt.show()
        return corr_to_plot