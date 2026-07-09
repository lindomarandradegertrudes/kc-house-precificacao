"""
plots.py

Funções de visualização usadas na EDA (Fase 1) e na avaliação do
modelo (Fase 6). Cada função salva a figura em outputs/figures/.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_histograma_price(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=50, kde=True)
    plt.title("Distribuição do preço dos imóveis")
    plt.xlabel("Preço (US$)")
    plt.savefig(FIGURES_DIR / "hist_price.png")
    plt.show()


def plot_dispersao_sqftliving_price(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["sqft_living"], y=df["price"], alpha=0.3)
    plt.title("Área construída x Preço")
    plt.savefig(FIGURES_DIR / "scatter_sqftliving_price.png")
    plt.show()


def plot_heatmap_correlacao(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Mapa de calor - Correlação de Pearson")
    plt.savefig(FIGURES_DIR / "heatmap_correlacao.png")
    plt.show()


def plot_boxplots_outliers(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(x=df["price"], ax=axes[0])
    sns.boxplot(x=df["bedrooms"], ax=axes[1])
    sns.boxplot(x=df["sqft_living"], ax=axes[2])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_outliers.png")
    plt.show()


def plot_real_vs_previsto(y_test, pred_test) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, pred_test, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", linewidth=2
    )
    plt.xlabel("Valor real (US$)")
    plt.ylabel("Valor previsto (US$)")
    plt.title("Valores reais vs. valores previstos")
    plt.savefig(FIGURES_DIR / "real_vs_previsto.png")
    plt.show()


def plot_residuos(y_test, pred_test) -> None:
    residuos = y_test - pred_test
    plt.figure(figsize=(8, 5))
    sns.histplot(residuos, bins=50, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Resíduo (valor real - valor previsto)")
    plt.title("Distribuição dos resíduos")
    plt.savefig(FIGURES_DIR / "residuos.png")
    plt.show()
