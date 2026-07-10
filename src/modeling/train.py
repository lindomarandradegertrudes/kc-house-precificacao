"""
train.py

Funções de preparação para modelagem (Fase 4), treino (Fase 5) e
avaliação/versionamento do modelo (Fase 6).
"""

import json
import pickle
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FEATURES, TARGET, RANDOM_STATE, TEST_SIZE, MODELS_DIR


def preparar_dados(df: pd.DataFrame):
    """
    Seleciona as variáveis preditoras (X) e a variável-alvo (y),
    divide em treino/teste e aplica escalonamento (fit no treino,
    transform no teste, evitando vazamento de dados).
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def treinar_modelo(X_train_scaled, y_train) -> LinearRegression:
    """Treina um modelo de Regressão Linear (usado para VALIDAÇÃO,
    com split treino/teste)."""
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)
    return modelo


def treinar_modelo_final(df: pd.DataFrame):
    """
    Treina o modelo FINAL (o que será salvo em models/v1/) usando 100%
    dos dados disponíveis, sem separar treino/teste.

    Diferente de treinar_modelo(), que serve apenas para VALIDAR o
    desempenho do modelo (métricas honestas sobre dados não vistos),
    este modelo final aproveita toda a informação disponível, já que
    não será mais avaliado após o treino - as métricas de referência
    continuam sendo as obtidas na validação com treino/teste.
    """
    X = df[FEATURES]
    y = df[TARGET]

    scaler_final = StandardScaler()
    X_scaled_full = scaler_final.fit_transform(X)

    modelo_final = LinearRegression()
    modelo_final.fit(X_scaled_full, y)

    return modelo_final, scaler_final


def diagnosticar_overfitting(modelo, X_train_scaled, y_train, X_test_scaled, y_test) -> dict:
    """
    Compara o RMSE de treino e teste para diagnosticar overfitting.
    Uma diferença pequena indica boa capacidade de generalização.
    """
    pred_train = modelo.predict(X_train_scaled)
    pred_test = modelo.predict(X_test_scaled)

    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    return {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "diferenca": abs(rmse_test - rmse_train),
    }


def avaliar_modelo(modelo, X_test_scaled, y_test) -> dict:
    """Calcula MAE, MSE, RMSE e R² no conjunto de teste."""
    pred_test = modelo.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, pred_test)
    mse = mean_squared_error(y_test, pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred_test)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "pred_test": pred_test}


def salvar_modelo(modelo_final, metricas: dict, versao: str = "v1") -> None:
    """
    Versiona o modelo FINAL (treinado na base completa, via
    treinar_modelo_final) e as métricas de VALIDAÇÃO (obtidas via
    avaliar_modelo, com split treino/teste) na pasta models/<versao>/,
    seguindo o padrão obrigatório da Fase 6.
    """
    pasta_versao = MODELS_DIR / versao
    pasta_versao.mkdir(parents=True, exist_ok=True)

    with open(pasta_versao / f"modelo_regressao_{versao}.pkl", "wb") as f:
        pickle.dump(modelo_final, f)

    metricas_salvas = {
        "MAE": metricas["MAE"],
        "MSE": metricas["MSE"],
        "RMSE": metricas["RMSE"],
        "R2": metricas["R2"],
        "data_treinamento": str(datetime.datetime.now()),
        "variaveis_preditoras": FEATURES,
        "observacao": (
            "Métricas obtidas com modelo validado via split treino/teste. "
            "O modelo .pkl salvo foi retreinado com 100% dos dados disponíveis."
        ),
    }

    with open(pasta_versao / f"metricas_{versao}.json", "w") as f:
        json.dump(metricas_salvas, f, indent=4)

    print(f"Modelo final (base completa) e métricas de validação salvos com sucesso em models/{versao}/")