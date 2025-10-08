import joblib as jl

import src.config as config
import src.evaluate as evaluate
import src.train_cv as train_cv
from src.preprocess import carregar_dados, preprocessar_dados, separar_dados


def salvar_modelo(modelo, caminho):
    caminho.parent.mkdir(exist_ok=True, parents=True)
    jl.dump(modelo, caminho)


def carregar_modelo(caminho):
    modelo = jl.load(caminho)

    return modelo


def criar_treinar_avaliar():
    features, target = carregar_dados()

    x_treino, x_teste, y_treino, y_teste = separar_dados(features, target)

    x_treino, y_treino = preprocessar_dados(x_treino, y_treino)
    x_teste, y_teste = preprocessar_dados(x_teste, y_teste)

    modelo = train_cv.treinar_e_cv(x_treino, y_treino)

    salvar_modelo(modelo, config.CAMINHO_MODELO)

    # Previsao
    y_pred = modelo.predict(x_teste)
    # Chance de cada possibilidade de previsao
    y_prob = modelo.predict_proba(x_teste)[:, 1]

    evaluate.avaliar_modelo(x_teste, y_teste, y_pred, y_prob)


def carregar_e_plotar():
    features, target = carregar_dados()

    x_treino, x_teste, y_treino, y_teste = separar_dados(features, target)

    x_treino, y_treino = preprocessar_dados(x_treino, y_treino)
    x_teste, y_teste = preprocessar_dados(x_teste, y_teste)

    modelo = carregar_modelo(config.CAMINHO_MODELO)

    # Previsao
    y_pred = modelo.predict(x_teste)
    # Chance de cada possibilidade de previsao
    y_prob = modelo.predict_proba(x_teste)[:, 1]

    evaluate.plotar_modelo(y_teste, y_pred, y_prob)
