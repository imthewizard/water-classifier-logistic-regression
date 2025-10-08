from pathlib import Path

# Valores padrão
RANDOM_STATE = 42
TAMANHO_TESTE = 0.2
STRATIFIED_K_FOLD_SPLITS = 10

# Caminhos padrão
CAMINHO_DADOS = Path("data/raw/water.csv").resolve()
CAMINHO_MODELO = Path("artifacts/best.pkl").resolve()

# Pastas padrão
PASTA_RESULTADO = Path("reports/").resolve()
PASTA_FIGURAS = Path("figures/").resolve()


HIPERPARAMETROS_BUSCA = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Forca da regularizacao
    "penalty": ["l1", "l2"],  # Tipo de regularizacao
    "solver": ["liblinear", "saga"],  # Algoritmo de otimizacao pra encontrar parametros
}

LISTA_FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

VARIAVEL_TARGET = "Potability"
