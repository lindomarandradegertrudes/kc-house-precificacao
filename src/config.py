"""
config.py

Centraliza caminhos de arquivos e parâmetros usados em todo o projeto.
Assim, se algo mudar de lugar, só precisa ajustar aqui.
"""

from pathlib import Path

# Raiz do projeto (uma pasta acima de src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Caminhos de dados
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "kc_house_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "kc_house_data_processed.csv"
FINAL_DATA_PATH = BASE_DIR / "data" / "final" / "kc_house_data_final.csv"

# Caminhos de saída
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MODELS_DIR = BASE_DIR / "models"

# Parâmetros de modelagem
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Variáveis preditoras finais (após remoção por multicolinearidade)
FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade",
    "idade_imovel", "foi_reformado", "lat", "long"
]

TARGET = "price"
