from sklearn.linear_model import LogisticRegression

import src.config as config


def criar_logistic_regression():
    modelo = LogisticRegression(random_state=config.RANDOM_STATE)

    return modelo
