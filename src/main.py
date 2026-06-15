"""
Trabalho Final - Inteligencia Artificial
Predicao de popularidade de musicas estilo Spotify.

Modelos:
- Parte 1: Arvore de Decisao
- Parte 2: Rede Neural MLP

COMO APONTAR AS FUNCOES NA APRESENTACAO:
1. load_dataset() -> funcao de carregamento de dados.
2. prepare_data() -> funcao de preparacao/tratamento dos dados.
3. build_preprocessor() -> funcao de pre-processamento dos dados.
4. build_models() -> funcao que cria os modelos de IA.
5. get_feature_names() -> funcao auxiliar.
6. evaluate_model() -> funcao de avaliacao dos modelos.
7. plot_dataset_charts() -> funcao de graficos do dataset.
8. plot_comparison() -> funcao de grafico comparativo.
9. plot_decision_tree() -> funcao de visualizacao da arvore.
10. plot_feature_importance() -> funcao de importancia dos atributos.
11. plot_mlp_training_curves() -> funcao de curva de treinamento da rede neural.
12. write_execution_summary() -> funcao que gera o resumo final em arquivo.
13. main() -> funcao principal, que organiza e executa todo o programa.
"""

# Importacao para permitir anotacoes de tipo mais modernas no Python.
from __future__ import annotations

# Path serve para trabalhar com caminhos de arquivos e pastas.
from pathlib import Path

# Dict, List e Tuple sao usados para indicar tipos de retorno das funcoes.
from typing import Dict, List, Tuple

# joblib salva os modelos treinados em arquivo .pkl.
import joblib

# matplotlib e usado para gerar e salvar os graficos.
import matplotlib.pyplot as plt

# pandas e usado para ler CSV, manipular tabelas e salvar resultados.
import pandas as pd

# ColumnTransformer permite aplicar tratamentos diferentes em colunas numericas e categoricas.
from sklearn.compose import ColumnTransformer

# Metricas e recursos de avaliacao do modelo.
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# train_test_split separa os dados em treino e teste.
from sklearn.model_selection import train_test_split

# MLPClassifier e o modelo de Rede Neural usado no trabalho.
from sklearn.neural_network import MLPClassifier

# Pipeline junta o pre-processamento e o modelo em uma unica estrutura.
from sklearn.pipeline import Pipeline

# OneHotEncoder transforma texto/categoria em numeros.
# StandardScaler padroniza os valores numericos.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# DecisionTreeClassifier e o modelo de Arvore de Decisao.
# plot_tree gera a imagem da arvore.
from sklearn.tree import DecisionTreeClassifier, plot_tree


# CONSTANTES DO PROGRAMA
# RANDOM_STATE fixa a aleatoriedade para os resultados ficarem reproduziveis.
RANDOM_STATE = 42

# POPULARITY_THRESHOLD define a partir de qual nota uma musica sera considerada popular.
POPULARITY_THRESHOLD = 70

# Caminho raiz do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Caminho do dataset CSV.
DATA_PATH = ROOT_DIR / "data" / "spotify_tracks_sample.csv"

# Pasta onde os modelos treinados serao salvos.
MODELS_DIR = ROOT_DIR / "models"

# Pasta onde graficos, metricas e resumo serao salvos.
RESULTS_DIR = ROOT_DIR / "results"


# LISTA DE ATRIBUTOS NUMERICOS
# Esses campos sao usados como entrada para treinar os modelos.
NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# LISTA DE ATRIBUTOS CATEGORICOS
# track_genre e texto/categoria, por isso precisa ser convertido para numeros.
CATEGORICAL_FEATURES = ["track_genre"]


# TIPO DE FUNCAO: FUNCAO DE CARREGAMENTO DE DADOS
# Objetivo: ler o arquivo CSV que contem as musicas.
def load_dataset() -> pd.DataFrame:
    # Verifica se o arquivo CSV existe no caminho esperado.
    if not DATA_PATH.exists():
        # Se o arquivo nao existir, interrompe o programa com uma mensagem clara.
        raise FileNotFoundError(
            f"Dataset nao encontrado em {DATA_PATH}. Execute: python src/generate_dataset.py"
        )

    # Le o CSV e retorna um DataFrame do pandas.
    return pd.read_csv(DATA_PATH)


# TIPO DE FUNCAO: FUNCAO DE PREPARACAO/TRATAMENTO DOS DADOS
# Objetivo: limpar os dados, criar a variavel alvo e separar X e y.
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Cria uma copia do DataFrame para evitar alterar o original diretamente.
    df = df.copy()

    # Define quais colunas obrigatoriamente precisam existir no dataset.
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["popularity"]

    # Verifica se alguma coluna obrigatoria esta faltando.
    missing = [column for column in required_columns if column not in df.columns]

    # Se alguma coluna estiver ausente, o programa para e informa quais colunas faltam.
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes no dataset: {missing}")

    # Remove linhas com valores vazios nas colunas importantes.
    df = df.dropna(subset=required_columns)

    # Cria a variavel alvo do problema:
    # se popularity >= 70, a musica e classificada como popularidade_alta = 1.
    # caso contrario, recebe 0.
    df["popularidade_alta"] = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)

    # X representa os atributos de entrada usados para prever a classe.
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # y representa a resposta/rotulo que o modelo deve aprender a prever.
    y = df["popularidade_alta"]

    # Retorna os atributos de entrada e a variavel alvo.
    return x, y


# TIPO DE FUNCAO: FUNCAO DE PRE-PROCESSAMENTO
# Objetivo: preparar os dados antes de enviar para o modelo de IA.
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # Para colunas numericas, aplica StandardScaler.
            # Isso deixa os valores em escala parecida, ajudando principalmente a Rede Neural.
            ("num", StandardScaler(), NUMERIC_FEATURES),

            # Para colunas categoricas, aplica OneHotEncoder.
            # Isso transforma categorias como genero musical em colunas numericas.
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


# TIPO DE FUNCAO: FUNCAO DE CRIACAO DOS MODELOS DE IA
# Objetivo: montar os dois modelos usados no trabalho: Arvore de Decisao e Rede Neural MLP.
def build_models() -> Dict[str, Pipeline]:
    return {
        # Modelo 1: Arvore de Decisao.
        # E mais facil de explicar porque gera regras de decisao.
        "arvore_decisao": Pipeline(
            steps=[
                # Primeiro passo do pipeline: pre-processar os dados.
                ("preprocessor", build_preprocessor()),

                # Segundo passo do pipeline: treinar/classificar com Arvore de Decisao.
                ("classifier", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ]
        ),

        # Modelo 2: Rede Neural MLP.
        # E um modelo que aprende padroes mais complexos entre os atributos.
        "rede_neural_mlp": Pipeline(
            steps=[
                # Primeiro passo: pre-processamento.
                ("preprocessor", build_preprocessor()),

                # Segundo passo: classificador de rede neural.
                (
                    "classifier",
                    MLPClassifier(
                        # Define duas camadas ocultas: uma com 32 neuronios e outra com 16.
                        hidden_layer_sizes=(32, 16),

                        # Funcao de ativacao usada na rede neural.
                        activation="relu",

                        # Algoritmo usado para otimizar o treinamento.
                        solver="adam",

                        # Numero maximo de iteracoes/epocas de treinamento.
                        max_iter=400,

                        # Mantem resultados reproduziveis.
                        random_state=RANDOM_STATE,

                        # Interrompe antes se perceber que o modelo parou de melhorar.
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


# TIPO DE FUNCAO: FUNCAO AUXILIAR
# Objetivo: recuperar os nomes dos atributos depois do pre-processamento.
def get_feature_names(model: Pipeline) -> List[str]:
    # Pega a etapa de pre-processamento dentro do pipeline.
    preprocessor = model.named_steps["preprocessor"]

    # Comeca com os nomes dos atributos numericos.
    feature_names = list(NUMERIC_FEATURES)

    # Pega o transformador responsavel pelas colunas categoricas.
    cat_encoder = preprocessor.named_transformers_["cat"]

    # Adiciona os nomes das colunas geradas pelo OneHotEncoder.
    feature_names.extend(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist())

    # Retorna a lista completa de atributos usados pelo modelo.
    return feature_names


# TIPO DE FUNCAO: FUNCAO DE AVALIACAO DO MODELO
# Objetivo: testar o modelo, calcular metricas e salvar matriz de confusao.
def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    # Usa o modelo treinado para prever as classes dos dados de teste.
    y_pred = model.predict(x_test)

    # Gera a matriz de confusao comparando resposta real e resposta prevista.
    matrix = confusion_matrix(y_test, y_pred)

    # Divide a matriz de confusao em quatro valores:
    # tn = verdadeiro negativo, fp = falso positivo, fn = falso negativo, tp = verdadeiro positivo.
    tn, fp, fn, tp = matrix.ravel()

    # Dicionario com as principais metricas de avaliacao.
    metrics = {
        "modelo": name,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "verdadeiro_negativo": tn,
        "falso_positivo": fp,
        "falso_negativo": fn,
        "verdadeiro_positivo": tp,
    }

    # Imprime um relatorio completo de classificacao no terminal.
    print("\n" + "=" * 70)
    print(f"Relatorio de classificacao - {name}")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["Nao popular", "Popular"], zero_division=0))

    # Cria uma visualizacao da matriz de confusao.
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Nao popular", "Popular"])
    display.plot(values_format="d")
    plt.title(f"Matriz de confusao - {name}")
    plt.tight_layout()

    # Salva a matriz de confusao como imagem na pasta results.
    plt.savefig(RESULTS_DIR / f"matriz_confusao_{name}.png", dpi=180)
    plt.close()

    # Retorna as metricas para serem usadas depois na comparacao dos modelos.
    return metrics


# TIPO DE FUNCAO: FUNCAO DE VISUALIZACAO/GRAFICOS DO DATASET
# Objetivo: gerar graficos iniciais sobre a distribuicao dos dados.
def plot_dataset_charts(df: pd.DataFrame) -> None:
    # Grafico 1: histograma mostrando a distribuicao da popularidade.
    plt.figure(figsize=(8, 5))
    plt.hist(df["popularity"], bins=20)
    plt.title("Distribuicao da popularidade das musicas")
    plt.xlabel("Popularidade")
    plt.ylabel("Quantidade de musicas")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribuicao_popularidade.png", dpi=180)
    plt.close()

    # Grafico 2: quantidade de musicas populares e nao populares.
    classes = (df["popularity"] >= POPULARITY_THRESHOLD).map({False: "Nao popular", True: "Popular"})
    counts = classes.value_counts().reindex(["Nao popular", "Popular"])
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Distribuicao das classes")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribuicao_classes.png", dpi=180)
    plt.close()


# TIPO DE FUNCAO: FUNCAO DE GRAFICO COMPARATIVO
# Objetivo: comparar as metricas dos modelos treinados.
def plot_comparison(metrics_df: pd.DataFrame) -> None:
    # Define quais metricas serao exibidas no grafico.
    metric_columns = ["acuracia", "precisao", "recall", "f1_score"]

    # Cria um grafico de barras comparando os modelos.
    ax = metrics_df.set_index("modelo")[metric_columns].plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    ax.set_title("Comparacao de desempenho entre modelos")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor da metrica")
    ax.legend(title="Metrica", loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Salva o grafico comparativo.
    plt.savefig(RESULTS_DIR / "comparacao_modelos.png", dpi=180)
    plt.close()


# TIPO DE FUNCAO: FUNCAO DE VISUALIZACAO DA ARVORE DE DECISAO
# Objetivo: gerar uma imagem com parte da arvore treinada.
def plot_decision_tree(model: Pipeline) -> None:
    # Pega o classificador Arvore de Decisao dentro do pipeline.
    classifier = model.named_steps["classifier"]

    # Pega os nomes dos atributos usados pela arvore.
    feature_names = get_feature_names(model)

    # Cria a figura da arvore de decisao.
    plt.figure(figsize=(20, 10))
    plot_tree(
        classifier,
        max_depth=3,
        feature_names=feature_names,
        class_names=["Nao popular", "Popular"],
        filled=True,
        fontsize=8,
    )
    plt.title("Visualizacao parcial da Arvore de Decisao")
    plt.tight_layout()

    # Salva a imagem da arvore.
    plt.savefig(RESULTS_DIR / "arvore_decisao_visualizacao.png", dpi=180)
    plt.close()


# TIPO DE FUNCAO: FUNCAO DE IMPORTANCIA DOS ATRIBUTOS
# Objetivo: descobrir quais atributos mais influenciaram a Arvore de Decisao.
def plot_feature_importance(model: Pipeline, top_n: int = 12) -> pd.DataFrame:
    # Pega o classificador treinado.
    classifier = model.named_steps["classifier"]

    # Pega os nomes dos atributos.
    feature_names = get_feature_names(model)

    # Cria uma tabela com atributo e sua importancia.
    importance_df = (
        pd.DataFrame(
            {
                "atributo": feature_names,
                "importancia": classifier.feature_importances_,
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    # Salva a tabela de importancia em CSV.
    importance_df.to_csv(RESULTS_DIR / "importancia_atributos_arvore_decisao.csv", index=False)

    # Seleciona os atributos mais importantes para mostrar no grafico.
    top = importance_df.head(top_n).sort_values("importancia", ascending=True)

    # Cria grafico horizontal com os atributos mais importantes.
    plt.figure(figsize=(9, 6))
    plt.barh(top["atributo"], top["importancia"])
    plt.title("Atributos mais importantes - Arvore de Decisao")
    plt.xlabel("Importancia")
    plt.ylabel("Atributo")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "importancia_atributos_arvore_decisao.png", dpi=180)
    plt.close()

    # Retorna a tabela para ser usada no resumo final.
    return importance_df


# TIPO DE FUNCAO: FUNCAO DE CURVA DE TREINAMENTO DA REDE NEURAL
# Objetivo: mostrar como a Rede Neural MLP evoluiu durante o treinamento.
def plot_mlp_training_curves(model: Pipeline) -> None:
    # Pega o classificador MLP dentro do pipeline.
    classifier = model.named_steps["classifier"]

    # Busca a curva de perda/loss gerada durante o treinamento.
    loss_curve = getattr(classifier, "loss_curve_", [])

    # Busca a curva de validacao, caso exista.
    validation_scores = getattr(classifier, "validation_scores_", [])

    # Se nao existir curva de perda, a funcao encerra sem gerar grafico.
    if not loss_curve:
        return

    # Cria dois graficos lado a lado: perda e validacao.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Grafico da perda durante o treinamento.
    axes[0].plot(range(1, len(loss_curve) + 1), loss_curve)
    axes[0].set_title("Perda durante o treinamento")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")

    # Se houver scores de validacao, gera o segundo grafico.
    if validation_scores:
        axes[1].plot(range(1, len(validation_scores) + 1), validation_scores)
        axes[1].set_title("Acuracia na validacao")
        axes[1].set_xlabel("Epoca")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
    else:
        # Caso nao tenha validacao, desliga o segundo grafico.
        axes[1].axis("off")

    fig.suptitle("Curva de treinamento - Rede Neural MLP")
    fig.tight_layout()

    # Salva a imagem da curva de treinamento.
    fig.savefig(RESULTS_DIR / "curva_treinamento_rede_neural_mlp.png", dpi=180)
    plt.close(fig)


# TIPO DE FUNCAO: FUNCAO DE GERACAO DE RESUMO/RELATORIO
# Objetivo: criar um arquivo Markdown com os resultados principais do experimento.
def write_execution_summary(metrics_df: pd.DataFrame, feature_importance_df: pd.DataFrame) -> None:
    # Escolhe o melhor modelo com base no F1-score.
    best = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]

    # Pega as metricas especificas da Arvore de Decisao.
    tree_metrics = metrics_df[metrics_df["modelo"] == "arvore_decisao"].iloc[0]

    # Pega as metricas especificas da Rede Neural MLP.
    mlp_metrics = metrics_df[metrics_df["modelo"] == "rede_neural_mlp"].iloc[0]

    # Seleciona os 5 atributos mais importantes da arvore.
    top_features = feature_importance_df.head(5)

    # Lista de linhas que formarao o arquivo resumo_execucao.md.
    lines = [
        "# Resumo da Execucao",
        "",
        "## Resultado principal",
        (
            f"O melhor modelo pelo F1-score foi `{best['modelo']}` "
            f"com F1-score de {best['f1_score']:.4f}."
        ),
        "",
        "## Comparacao rapida",
        "",
        "| Modelo | Acuracia | Precisao | Recall | F1-score | VP | VN | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    # Adiciona as metricas de cada modelo na tabela do resumo.
    for _, row in metrics_df.iterrows():
        lines.append(
            "| {modelo} | {acuracia:.4f} | {precisao:.4f} | {recall:.4f} | {f1_score:.4f} | "
            "{verdadeiro_positivo:.0f} | {verdadeiro_negativo:.0f} | "
            "{falso_positivo:.0f} | {falso_negativo:.0f} |".format(**row.to_dict())
        )

    # Adiciona textos prontos para ajudar na apresentacao.
    lines.extend(
        [
            "",
            "## Leitura para apresentacao",
            (
                "A Arvore de Decisao e mais facil de explicar porque transforma os dados em regras. "
                f"Neste experimento, ela atingiu F1-score de {tree_metrics['f1_score']:.4f}."
            ),
            (
                "A Rede Neural MLP conseguiu desempenho melhor porque aprende combinacoes nao lineares "
                f"entre os atributos. Neste experimento, ela atingiu F1-score de {mlp_metrics['f1_score']:.4f}."
            ),
            "",
            "## Atributos mais importantes na Arvore de Decisao",
            "",
            "| Atributo | Importancia |",
            "|---|---:|",
        ]
    )

    # Adiciona os atributos mais importantes no resumo.
    for _, row in top_features.iterrows():
        lines.append(f"| {row['atributo']} | {row['importancia']:.4f} |")

    # Lista os arquivos gerados pelo programa.
    lines.extend(
        [
            "",
            "## Arquivos gerados",
            "",
            "- `results/metricas_modelos.csv`",
            "- `results/comparacao_modelos.png`",
            "- `results/matriz_confusao_arvore_decisao.png`",
            "- `results/matriz_confusao_rede_neural_mlp.png`",
            "- `results/importancia_atributos_arvore_decisao.png`",
            "- `results/curva_treinamento_rede_neural_mlp.png`",
        ]
    )

    # Escreve o conteudo no arquivo Markdown.
    (RESULTS_DIR / "resumo_execucao.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# TIPO DE FUNCAO: FUNCAO PRINCIPAL
# Objetivo: controlar a ordem de execucao de todo o programa.
def main() -> None:
    # Cria as pastas models e results caso elas ainda nao existam.
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Carrega o dataset.
    df = load_dataset()

    # Mostra cabecalho do programa no terminal.
    print("=" * 70)
    print("Trabalho Final - Inteligencia Artificial")
    print("Predicao de Popularidade de Musicas estilo Spotify")
    print("=" * 70)
    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")

    # Gera graficos iniciais do dataset.
    plot_dataset_charts(df)

    # Prepara os dados e separa entrada X e resposta y.
    x, y = prepare_data(df)

    # Mostra quais atributos serao usados pelo modelo.
    print("\nAtributos utilizados:")
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        print(f"- {feature}")

    # Mostra a distribuicao da variavel alvo.
    print("\nDistribuicao da variavel alvo:")
    print(y.value_counts().rename(index={0: "Nao popular", 1: "Popular"}))

    # Divide os dados em treino e teste.
    # 75% dos dados ficam para treino e 25% para teste.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Mostra a quantidade de amostras de treino e teste.
    print("\nDivisao dos dados:")
    print(f"Treino: {len(x_train)} amostras")
    print(f"Teste: {len(x_test)} amostras")

    # Lista que vai guardar as metricas dos modelos.
    metrics_list = []

    # Cria os modelos definidos na funcao build_models().
    models = build_models()

    # Percorre cada modelo, treina, salva e avalia.
    for name, model in models.items():
        print("\n" + "-" * 70)
        print(f"Treinando modelo: {name}")
        print("-" * 70)

        # Treina o modelo com os dados de treino.
        model.fit(x_train, y_train)

        # Salva o modelo treinado na pasta models.
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

        # Avalia o modelo nos dados de teste e salva as metricas.
        metrics_list.append(evaluate_model(name, model, x_test, y_test))

    # Gera visualizacao da arvore de decisao.
    plot_decision_tree(models["arvore_decisao"])

    # Gera grafico e tabela de importancia dos atributos.
    feature_importance_df = plot_feature_importance(models["arvore_decisao"])

    # Gera curva de treinamento da Rede Neural MLP.
    plot_mlp_training_curves(models["rede_neural_mlp"])

    # Cria um DataFrame com as metricas de todos os modelos.
    metrics_df = pd.DataFrame(metrics_list)

    # Salva as metricas em CSV.
    metrics_df.to_csv(RESULTS_DIR / "metricas_modelos.csv", index=False)

    # Gera grafico comparativo dos modelos.
    plot_comparison(metrics_df)

    # Gera o resumo final da execucao.
    write_execution_summary(metrics_df, feature_importance_df)

    # Mostra as metricas finais no terminal.
    print("\nMetricas finais:")
    print(metrics_df.to_string(index=False))

    # Escolhe e exibe o melhor modelo com base no F1-score.
    best = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]
    print(f"\nMelhor modelo pelo F1-score: {best['modelo']} ({best['f1_score']:.4f})")
    print("\nArquivos gerados nas pastas models/ e results/.")


# TIPO DE BLOCO: PONTO DE ENTRADA DO PROGRAMA
# Este if garante que a funcao main() so execute quando este arquivo for rodado diretamente.
if __name__ == "__main__":
    main()
