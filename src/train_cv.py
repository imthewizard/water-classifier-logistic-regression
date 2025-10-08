import argparse
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold

import src.config as config
import src.evaluate as evaluate
import src.utils as utils
from src.model import criar_logistic_regression


def treinar_e_cv(x_treino, y_treino):
    modelo = criar_logistic_regression()

    # Busca por hiperparametros
    skf = StratifiedKFold(n_splits=config.STRATIFIED_K_FOLD_SPLITS)
    grid = GridSearchCV(
        modelo, config.HIPERPARAMETROS_BUSCA, cv=skf, scoring="accuracy", n_jobs=-1
    )

    # Treina o modelo
    grid.fit(x_treino, y_treino)

    melhor_modelo = grid.best_estimator_

    return melhor_modelo


if __name__ == "__main__":
    # Argumentos CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Caminho dos dados .csv", type=str)
    parser.add_argument("--out", help="Pasta do resultado", type=str)
    parser.add_argument(
        "--k", help="Quantidade de splits do stratified k-fold", type=int
    )
    parser.add_argument("--seed", help="Random state", type=int)

    args = parser.parse_args()
    if args.data:
        config.CAMINHO_DADOS = Path(args.data).resolve()
        if not config.CAMINHO_DADOS.is_file():
            print("Caminho dos dados incorreto")
            exit()
    if args.out:
        config.PASTA_RESULTADO = Path(args.out).resolve()
    if args.k:
        config.STRATIFIED_K_FOLD_SPLITS = args.k
    if args.seed:
        config.RANDOM_STATE = args.seed

    utils.criar_treinar_avaliar()
