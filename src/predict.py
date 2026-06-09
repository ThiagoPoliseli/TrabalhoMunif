"""Exemplos de predicao usando o modelo treinado.

Use este arquivo na apresentacao para demonstrar como o modelo recebe
atributos de uma musica e devolve a classe prevista.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "rede_neural_mlp.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo nao encontrado. Execute primeiro: python src/main.py")

    model = joblib.load(MODEL_PATH)
    samples = pd.DataFrame(
        [
            {
                "descricao": "Musica pop dancante e energetica",
                "danceability": 0.82,
                "energy": 0.76,
                "loudness": -4.8,
                "speechiness": 0.05,
                "acousticness": 0.18,
                "instrumentalness": 0.01,
                "liveness": 0.12,
                "valence": 0.74,
                "tempo": 124.0,
                "duration_ms": 205000,
                "track_genre": "pop",
            },
            {
                "descricao": "Musica indie mais acustica e instrumental",
                "danceability": 0.38,
                "energy": 0.34,
                "loudness": -13.5,
                "speechiness": 0.04,
                "acousticness": 0.72,
                "instrumentalness": 0.42,
                "liveness": 0.19,
                "valence": 0.31,
                "tempo": 92.0,
                "duration_ms": 242000,
                "track_genre": "indie",
            },
            {
                "descricao": "Funk com batida forte e alta dancabilidade",
                "danceability": 0.88,
                "energy": 0.83,
                "loudness": -3.9,
                "speechiness": 0.11,
                "acousticness": 0.08,
                "instrumentalness": 0.02,
                "liveness": 0.10,
                "valence": 0.69,
                "tempo": 132.0,
                "duration_ms": 198000,
                "track_genre": "funk",
            },
        ]
    )

    features = samples.drop(columns=["descricao"])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    for description, prediction, probability in zip(samples["descricao"], predictions, probabilities):
        label = "Popular" if prediction == 1 else "Nao popular"
        print("-" * 70)
        print(description)
        print(f"Predicao: {label}")
        print(f"Probabilidade de ser popular: {probability:.2%}")


if __name__ == "__main__":
    main()
