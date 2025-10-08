import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.config as config


def carregar_dados():
    # Carrega o arquivo .csv
    dados = pd.read_csv(config.CAMINHO_DADOS)

    # Colunas de features e coluna de target
    features = dados[config.LISTA_FEATURES]
    target = dados[config.VARIAVEL_TARGET]

    return features, target


def preprocessar_dados(x, y):
    # Imputacao por mediana para valores ausentes
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x)

    # Checar o quanto desvia da media e filtrar
    z_scores = abs(zscore(x))
    mask = (z_scores < 1.75).all(axis=1)
    x = x[mask]
    y = y[mask]

    # Padroniza os dados, mesma escala
    scaler = StandardScaler()
    x_dimensionado = scaler.fit_transform(x)
    return x_dimensionado, y


def separar_dados(features, target):
    # Separação em teste/treino
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        features,
        target,
        test_size=config.TAMANHO_TESTE,
        random_state=config.RANDOM_STATE,
    )

    return x_treino, x_teste, y_treino, y_teste
