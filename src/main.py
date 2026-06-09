"""Trabalho Final - Inteligencia Artificial
Predicao de popularidade de musicas estilo Spotify.

Modelos:
- Parte 1: Arvore de Decisao
- Parte 2: Rede Neural MLP
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
POPULARITY_THRESHOLD = 70

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "spotify_tracks_sample.csv"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

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
CATEGORICAL_FEATURES = ["track_genre"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em {DATA_PATH}. Execute: python src/generate_dataset.py"
        )
    return pd.read_csv(DATA_PATH)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["popularity"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes no dataset: {missing}")

    df = df.dropna(subset=required_columns)
    df["popularidade_alta"] = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["popularidade_alta"]
    return x, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> Dict[str, Pipeline]:
    return {
        "arvore_decisao": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ]
        ),
        "rede_neural_mlp": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        max_iter=400,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def get_feature_names(model: Pipeline) -> List[str]:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = list(NUMERIC_FEATURES)
    cat_encoder = preprocessor.named_transformers_["cat"]
    feature_names.extend(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist())
    return feature_names


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matrix.ravel()
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

    print("\n" + "=" * 70)
    print(f"Relatorio de classificacao - {name}")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["Nao popular", "Popular"], zero_division=0))

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Nao popular", "Popular"])
    display.plot(values_format="d")
    plt.title(f"Matriz de confusao - {name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"matriz_confusao_{name}.png", dpi=180)
    plt.close()

    return metrics


def plot_dataset_charts(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["popularity"], bins=20)
    plt.title("Distribuicao da popularidade das musicas")
    plt.xlabel("Popularidade")
    plt.ylabel("Quantidade de musicas")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribuicao_popularidade.png", dpi=180)
    plt.close()

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


def plot_comparison(metrics_df: pd.DataFrame) -> None:
    metric_columns = ["acuracia", "precisao", "recall", "f1_score"]
    ax = metrics_df.set_index("modelo")[metric_columns].plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    ax.set_title("Comparacao de desempenho entre modelos")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor da metrica")
    ax.legend(title="Metrica", loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparacao_modelos.png", dpi=180)
    plt.close()


def plot_decision_tree(model: Pipeline) -> None:
    classifier = model.named_steps["classifier"]
    feature_names = get_feature_names(model)

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
    plt.savefig(RESULTS_DIR / "arvore_decisao_visualizacao.png", dpi=180)
    plt.close()


def plot_feature_importance(model: Pipeline, top_n: int = 12) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    feature_names = get_feature_names(model)
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
    importance_df.to_csv(RESULTS_DIR / "importancia_atributos_arvore_decisao.csv", index=False)

    top = importance_df.head(top_n).sort_values("importancia", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top["atributo"], top["importancia"])
    plt.title("Atributos mais importantes - Arvore de Decisao")
    plt.xlabel("Importancia")
    plt.ylabel("Atributo")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "importancia_atributos_arvore_decisao.png", dpi=180)
    plt.close()
    return importance_df


def plot_mlp_training_curves(model: Pipeline) -> None:
    classifier = model.named_steps["classifier"]
    loss_curve = getattr(classifier, "loss_curve_", [])
    validation_scores = getattr(classifier, "validation_scores_", [])

    if not loss_curve:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(loss_curve) + 1), loss_curve)
    axes[0].set_title("Perda durante o treinamento")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")

    if validation_scores:
        axes[1].plot(range(1, len(validation_scores) + 1), validation_scores)
        axes[1].set_title("Acuracia na validacao")
        axes[1].set_xlabel("Epoca")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
    else:
        axes[1].axis("off")

    fig.suptitle("Curva de treinamento - Rede Neural MLP")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "curva_treinamento_rede_neural_mlp.png", dpi=180)
    plt.close(fig)


def write_execution_summary(metrics_df: pd.DataFrame, feature_importance_df: pd.DataFrame) -> None:
    best = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]
    tree_metrics = metrics_df[metrics_df["modelo"] == "arvore_decisao"].iloc[0]
    mlp_metrics = metrics_df[metrics_df["modelo"] == "rede_neural_mlp"].iloc[0]
    top_features = feature_importance_df.head(5)

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

    for _, row in metrics_df.iterrows():
        lines.append(
            "| {modelo} | {acuracia:.4f} | {precisao:.4f} | {recall:.4f} | {f1_score:.4f} | "
            "{verdadeiro_positivo:.0f} | {verdadeiro_negativo:.0f} | "
            "{falso_positivo:.0f} | {falso_negativo:.0f} |".format(**row.to_dict())
        )

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

    for _, row in top_features.iterrows():
        lines.append(f"| {row['atributo']} | {row['importancia']:.4f} |")

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

    (RESULTS_DIR / "resumo_execucao.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    print("=" * 70)
    print("Trabalho Final - Inteligencia Artificial")
    print("Predicao de Popularidade de Musicas estilo Spotify")
    print("=" * 70)
    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")

    plot_dataset_charts(df)
    x, y = prepare_data(df)
    print("\nAtributos utilizados:")
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        print(f"- {feature}")

    print("\nDistribuicao da variavel alvo:")
    print(y.value_counts().rename(index={0: "Nao popular", 1: "Popular"}))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("\nDivisao dos dados:")
    print(f"Treino: {len(x_train)} amostras")
    print(f"Teste: {len(x_test)} amostras")

    metrics_list = []
    models = build_models()
    for name, model in models.items():
        print("\n" + "-" * 70)
        print(f"Treinando modelo: {name}")
        print("-" * 70)
        model.fit(x_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        metrics_list.append(evaluate_model(name, model, x_test, y_test))

    plot_decision_tree(models["arvore_decisao"])
    feature_importance_df = plot_feature_importance(models["arvore_decisao"])
    plot_mlp_training_curves(models["rede_neural_mlp"])

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(RESULTS_DIR / "metricas_modelos.csv", index=False)
    plot_comparison(metrics_df)
    write_execution_summary(metrics_df, feature_importance_df)

    print("\nMetricas finais:")
    print(metrics_df.to_string(index=False))
    best = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]
    print(f"\nMelhor modelo pelo F1-score: {best['modelo']} ({best['f1_score']:.4f})")
    print("\nArquivos gerados nas pastas models/ e results/.")


if __name__ == "__main__":
    main()
