"""
dataset.py

Funções para carregar e salvar os dados em cada etapa do pipeline
(bruto -> processado -> final).
"""

import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FINAL_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    """Carrega o dataset original, sem qualquer tratamento."""
    return pd.read_csv(RAW_DATA_PATH)


def save_processed_data(df: pd.DataFrame) -> None:
    """Salva o dataset após limpeza e tratamento (Fase 2)."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def load_processed_data() -> pd.DataFrame:
    """Carrega o dataset já tratado."""
    return pd.read_csv(PROCESSED_DATA_PATH)


def save_final_data(df: pd.DataFrame) -> None:
    """Salva o recorte final de dados, já pronto para modelagem (Fase 4)."""
    FINAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_DATA_PATH, index=False)


def load_final_data() -> pd.DataFrame:
    """Carrega o recorte final de dados usado na modelagem."""
    return pd.read_csv(FINAL_DATA_PATH)
