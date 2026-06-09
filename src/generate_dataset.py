"""Gera um dataset tabular estilo Spotify para o trabalho final de IA.

O enunciado permite dataset criado pela equipe. Este script cria uma base
reprodutivel com atributos de audio parecidos com os usados em bases publicas
do Spotify, sem depender de download externo.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42


def clamp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(values, low), high)


def generate_spotify_style_dataset(n_samples: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    genres = np.array([
        "pop", "rock", "hip-hop", "electronic", "sertanejo", "funk", "indie", "latin", "r&b", "dance"
    ])
    genre = rng.choice(genres, size=n_samples, p=[0.18, 0.12, 0.13, 0.10, 0.10, 0.10, 0.08, 0.08, 0.06, 0.05])

    danceability = rng.beta(4.2, 2.4, n_samples)
    energy = rng.beta(3.2, 2.2, n_samples)
    acousticness = rng.beta(1.8, 4.0, n_samples)
    instrumentalness = rng.beta(0.8, 8.5, n_samples)
    liveness = rng.beta(1.5, 6.2, n_samples)
    speechiness = rng.beta(1.2, 8.0, n_samples)
    valence = rng.beta(2.5, 2.6, n_samples)
    tempo = clamp(rng.normal(121, 26, n_samples), 55, 210)
    duration_ms = clamp(rng.normal(215000, 52000, n_samples), 65000, 420000).astype(int)
    loudness = clamp(rng.normal(-8.2 + energy * 5.0, 3.2, n_samples), -35, 1)

    genre_bonus_map = {
        "pop": 10,
        "hip-hop": 8,
        "funk": 7,
        "sertanejo": 6,
        "electronic": 5,
        "latin": 5,
        "dance": 4,
        "rock": 2,
        "r&b": 2,
        "indie": -2,
    }
    genre_bonus = np.array([genre_bonus_map[g] for g in genre])

    noise = rng.normal(0, 10, n_samples)
    popularity_score = (
        27
        + 24 * danceability
        + 19 * energy
        + 12 * valence
        + 7 * (tempo > 95)
        + 5 * (tempo < 155)
        + 0.75 * loudness
        - 14 * acousticness
        - 16 * instrumentalness
        - 5 * liveness
        + genre_bonus
        + noise
    )
    popularity = clamp(popularity_score, 0, 100).round().astype(int)

    artists = [f"Artista {i:03d}" for i in rng.integers(1, 250, n_samples)]
    track_names = [f"Musica {i:05d}" for i in range(1, n_samples + 1)]
    track_ids = [f"trk_{i:05d}" for i in range(1, n_samples + 1)]

    return pd.DataFrame({
        "track_id": track_ids,
        "track_name": track_names,
        "artists": artists,
        "track_genre": genre,
        "popularity": popularity,
        "danceability": danceability.round(4),
        "energy": energy.round(4),
        "loudness": loudness.round(4),
        "speechiness": speechiness.round(4),
        "acousticness": acousticness.round(4),
        "instrumentalness": instrumentalness.round(4),
        "liveness": liveness.round(4),
        "valence": valence.round(4),
        "tempo": tempo.round(2),
        "duration_ms": duration_ms,
    })


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_spotify_style_dataset()
    output = out_dir / "spotify_tracks_sample.csv"
    df.to_csv(output, index=False)
    print(f"Dataset gerado em: {output}")
    print(f"Registros: {len(df)}")


if __name__ == "__main__":
    main()
