import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import src.config as config
import src.utils as utils


def plot_roc(y_teste, y_prob):
    roc_auc = roc_auc_score(y_teste, y_prob)
    fpr, tpr, thresholds = roc_curve(y_teste, y_prob)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.savefig(config.PASTA_FIGURAS / "roc.png")


def plot_precision_recall(y_teste, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_teste, y_prob)

    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.savefig(config.PASTA_FIGURAS / "precision-recall.png")


def plot_confusion_matrix(y_teste, y_pred):
    c_mat = confusion_matrix(y_teste, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=c_mat)
    display.plot()
    plt.savefig(config.PASTA_FIGURAS / "confusion-matrix.png")


def avaliar_modelo(x_teste, y_teste, y_pred, y_prob):
    # Matriz de confusão
    c_mat = confusion_matrix(y_teste, y_pred)
    print("Matriz de confusao:")
    print(c_mat)
    # Accuracy
    accuracy = accuracy_score(y_teste, y_pred)
    print("Accuracy:")
    print(accuracy)
    # F1
    f1 = f1_score(y_teste, y_pred)
    print("F1:")
    print(f1)
    # ROC AUC
    roc_auc = roc_auc_score(y_teste, y_prob)
    print("ROC AUC:")
    print(roc_auc)
    # Precision
    precision = precision_score(y_teste, y_pred)
    print("Precision:")
    print(precision)
    # Recall
    recall = recall_score(y_teste, y_pred)
    print("Recall:")
    print(recall)
    # PR-AUC
    pr_auc = average_precision_score(y_teste, y_prob)
    print("PR-AUC:")
    print(pr_auc)

    resultados = pd.DataFrame(
        {
            "Real": y_teste,
            "Previsao": y_pred,
        }
    )
    config.PASTA_RESULTADO.mkdir(exist_ok=True, parents=True)
    resultados.to_csv(config.PASTA_RESULTADO / "results.csv", index=False)


def plotar_modelo(y_teste, y_pred, y_prob):
    config.PASTA_FIGURAS.mkdir(exist_ok=True, parents=True)

    plot_roc(y_teste, y_prob)
    plot_precision_recall(y_teste, y_prob)
    plot_confusion_matrix(y_teste, y_pred)


if __name__ == "__main__":
    # Argumentos CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Caminho do modelo .pkl", type=str)
    parser.add_argument("--data", help="Caminho dos dados .csv", type=str)
    parser.add_argument("--out", help="Pasta das figuras", type=str)

    args = parser.parse_args()
    if args.model:
        config.CAMINHO_MODELO = Path(args.model).resolve()
        if not config.CAMINHO_MODELO.is_file():
            print("Caminho do modelo incorreto")
            exit()
    if args.data:
        config.CAMINHO_DADOS = Path(args.data).resolve()
        if not config.CAMINHO_DADOS.is_file():
            print("Caminho dos dados incorreto")
            exit()
    if args.out:
        config.PASTA_FIGURAS = Path(args.out).resolve()

    utils.carregar_e_plotar()
