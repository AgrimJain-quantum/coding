"""
recommender.py
--------------
Core recommendation engine for the Music Recommendation System.

Implements:
  - recommend()              Single-song content-based recommendations
  - recommend_from_history() Multi-song averaged-vector recommendations
  - explain_recommendation() Feature-level explanation of why a song was picked
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _build_similarity_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for all songs.

    Args:
        feature_matrix: Normalized feature matrix (n_songs × n_features).

    Returns:
        Symmetric similarity matrix (n_songs × n_songs), values in [-1, 1].
    """
    return cosine_similarity(feature_matrix)


def _find_song_index(song_name: str, df: pd.DataFrame) -> int:
    """
    Case-insensitive lookup for a song name in the DataFrame.

    Args:
        song_name: Song title to search for.
        df       : DataFrame containing a 'name' column.

    Returns:
        Integer row index of the matched song.

    Raises:
        ValueError: If the song is not found.
    """
    matches = df[df["name"].str.lower() == song_name.lower()]
    if matches.empty:
        raise ValueError(
            f"Song '{song_name}' not found in the dataset. "
            "Please choose a song from the dropdown."
        )
    return matches.index[0]


def _top_k_indices(similarity_scores: np.ndarray, exclude_indices: list[int], k: int) -> list[int]:
    """
    Return top-k most similar song indices, excluding seeds.

    Args:
        similarity_scores: 1-D array of similarity scores for all songs.
        exclude_indices  : Indices to skip (the seed songs themselves).
        k                : Number of recommendations to return.

    Returns:
        List of top-k indices sorted descending by similarity.
    """
    scores = similarity_scores.copy()
    scores[exclude_indices] = -999  # mask out seed songs
    top_k = np.argsort(scores)[::-1][:k]
    return top_k.tolist()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def recommend(
    song_name: str,
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend songs similar to a single input song.

    Uses cosine similarity on normalized audio features.

    Args:
        song_name     : Name of the seed song.
        df            : Full cleaned DataFrame.
        feature_matrix: Normalized feature matrix (n_songs × n_features).
        top_k         : Number of recommendations to return (default 5).

    Returns:
        DataFrame with top_k recommended songs and their similarity scores.
    """
    seed_idx   = _find_song_index(song_name, df)
    sim_matrix = _build_similarity_matrix(feature_matrix)
    sim_scores = sim_matrix[seed_idx]

    top_indices = _top_k_indices(sim_scores, exclude_indices=[seed_idx], k=top_k)

    results = df.iloc[top_indices].copy()
    results["similarity_score"] = sim_scores[top_indices]
    results = results.reset_index(drop=True)

    return results


def recommend_from_history(
    song_list: list[str],
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend songs based on a user's listening history (multiple seeds).

    Strategy: average the feature vectors of all seed songs, then find the
    globally most similar songs using cosine similarity against this centroid.

    Args:
        song_list     : List of song names the user has listened to.
        df            : Full cleaned DataFrame.
        feature_matrix: Normalized feature matrix (n_songs × n_features).
        top_k         : Number of recommendations to return (default 5).

    Returns:
        DataFrame with top_k recommended songs and their similarity scores.

    Raises:
        ValueError: If none of the provided song names are found in the dataset.
    """
    seed_indices = []
    not_found    = []

    for name in song_list:
        try:
            idx = _find_song_index(name, df)
            seed_indices.append(idx)
        except ValueError:
            not_found.append(name)

    if not seed_indices:
        raise ValueError(
            f"None of the provided songs were found: {not_found}. "
            "Please select songs from the dropdown list."
        )

    # Average feature vectors across all found seed songs
    seed_vectors   = feature_matrix[seed_indices]          # (n_seeds, n_features)
    centroid_vector = seed_vectors.mean(axis=0, keepdims=True)  # (1, n_features)

    # Compute cosine similarity of centroid against all songs
    sim_scores = cosine_similarity(centroid_vector, feature_matrix)[0]  # (n_songs,)

    top_indices = _top_k_indices(sim_scores, exclude_indices=seed_indices, k=top_k)

    results = df.iloc[top_indices].copy()
    results["similarity_score"] = sim_scores[top_indices]
    results = results.reset_index(drop=True)

    return results


def explain_recommendation(
    seed_name: str,
    recommended_name: str,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict]:
    """
    Generate a human-readable feature-level explanation for why a song was
    recommended.

    Computes the absolute difference for each feature between the seed and the
    recommended song (in original, un-normalized space), then ranks features
    by similarity (smallest difference = most similar).

    Args:
        seed_name        : Name of the seed song.
        recommended_name : Name of the recommended song.
        df               : Full cleaned DataFrame (un-normalized values).
        feature_columns  : List of feature column names used.

    Returns:
        List of dicts sorted by similarity descending, each containing:
          {feature, seed_value, rec_value, similarity_pct, description}
    """
    available_features = [f for f in feature_columns if f in df.columns]

    seed_row = df[df["name"].str.lower() == seed_name.lower()].iloc[0]
    rec_row  = df[df["name"].str.lower() == recommended_name.lower()].iloc[0]

    explanations = []
    for feat in available_features:
        seed_val = float(seed_row[feat])
        rec_val  = float(rec_row[feat])

        # Normalise difference against the column range for a 0-100% similarity
        col_range = df[feat].max() - df[feat].min()
        diff      = abs(seed_val - rec_val)
        similarity_pct = max(0.0, (1.0 - diff / (col_range + 1e-9))) * 100

        explanations.append({
            "feature":        feat,
            "seed_value":     round(seed_val, 3),
            "rec_value":      round(rec_val, 3),
            "similarity_pct": round(similarity_pct, 1),
            "description":    FEATURE_DESCRIPTIONS.get(feat, feat),
        })

    # Sort by similarity (most similar features first)
    explanations.sort(key=lambda x: x["similarity_pct"], reverse=True)
    return explanations


# ──────────────────────────────────────────────
# Quick sanity check when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    from preprocess import preprocess

    data_path = pathlib.Path(__file__).parent.parent / "data" / "songs.csv"
    df, matrix, feat_cols = preprocess(str(data_path))

    print("── Single-song recommendations ─────────────────")
    recs = recommend("Blinding Lights", df, matrix)
    print(recs[["name", "artist", "similarity_score"]])

    print("\n── History-based recommendations ───────────────")
    recs2 = recommend_from_history(["Blinding Lights", "Levitating"], df, matrix)
    print(recs2[["name", "artist", "similarity_score"]])

    print("\n── Feature explanation ──────────────────────────")
    exp = explain_recommendation("Blinding Lights", recs.iloc[0]["name"], df, feat_cols)
    for e in exp[:3]:
        print(f"  {e['feature']:20s} → {e['similarity_pct']}% similar")
