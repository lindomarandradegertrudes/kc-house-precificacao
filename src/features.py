"""
features.py

Funções de limpeza (Fase 2) e criação de colunas derivadas (Fase 3).
"""

import pandas as pd


def remover_outlier_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove o registro com 33 quartos (id 2402100895), identificado como
    erro de digitação: 1.620 sqft de área construída é incompatível com
    essa quantidade de quartos.
    """
    return df[df["id"] != 2402100895].copy()


def tratar_ausentes_sqft_above(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa os valores ausentes em sqft_above usando a relação exata:
    sqft_living = sqft_above + sqft_basement.
    """
    df = df.copy()
    df["sqft_above"] = df["sqft_above"].fillna(
        df["sqft_living"] - df["sqft_basement"]
    )
    return df


def criar_colunas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria as colunas de Feature Engineering (Fase 3):
    - idade_imovel: idade do imóvel no momento da venda (nunca negativa;
      casos de venda "na planta" são ajustados para 0).
    - foi_reformado: indicador binário de reforma.
    - preco_por_m2: apenas para leitura/EDA, não deve ser usada como
      variável preditora (vazamento de dados).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ano_venda"] = df["date"].dt.year
    df["idade_imovel"] = (df["ano_venda"] - df["yr_built"]).clip(lower=0)
    df["foi_reformado"] = (df["yr_renovated"] > 0).astype(int)
    df["preco_por_m2"] = df["price"] / df["sqft_living"]
    return df


def limpar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de limpeza completo (Fase 2 + Fase 3), aplicando todas as
    funções acima na ordem correta.
    """
    df = remover_outlier_bedrooms(df)
    df = tratar_ausentes_sqft_above(df)
    df = criar_colunas_derivadas(df)
    return df
