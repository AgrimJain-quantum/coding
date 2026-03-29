"""
preprocess.py
-------------
Handles all data loading, cleaning, and feature normalization
for the Music Recommendation System.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
# Audio features used for similarity computation
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "tempo",
    "valence",
    "loudness",
    "acousticness",
    "instrumentalness",
]

# Human-readable descriptions for each feature (used in UI explanations)
FEATURE_DESCRIPTIONS = {
    "danceability":     "How suitable the track is for dancing",
    "energy":           "Perceptual measure of intensity and activity",
    "tempo":            "Overall estimated tempo (BPM)",
    "valence":          "Musical positiveness / happiness conveyed",
    "loudness":         "Overall loudness in decibels (dB)",
    "acousticness":     "Confidence the track is acoustic",
    "instrumentalness": "Predicts whether the track has no vocals",
}


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the songs CSV file into a DataFrame.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place songs.csv inside the data/ directory."
        )
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and basic sanity checks.

    Strategy:
      - Numeric features → fill NaN with column median (robust to outliers)
      - String columns   → fill NaN with 'Unknown'
      - Drop duplicate song names (keep first occurrence)

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Fill numeric missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with placeholder
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in string_cols:
        df[col] = df[col].fillna("Unknown")

    # Remove duplicate song names (case-insensitive)
    df["name_lower"] = df["name"].str.lower().str.strip()
    df.drop_duplicates(subset="name_lower", keep="first", inplace=True)
    df.drop(columns=["name_lower"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the audio feature columns that exist in the DataFrame.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame containing only the available feature columns.
    """
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    return df[available]


def normalize_features(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Normalize feature columns using StandardScaler (zero mean, unit variance).
    This ensures no single feature dominates the cosine similarity calculation.

    Args:
        feature_df: DataFrame of raw feature values.

    Returns:
        NumPy array of scaled features, shape (n_songs, n_features).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    return scaled


def preprocess(filepath: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Full preprocessing pipeline: load → clean → extract → normalize.

    Args:
        filepath: Path to the CSV dataset.

    Returns:
        Tuple of:
          - df              : Cleaned full DataFrame (with name, artist, genre, etc.)
          - feature_matrix  : Normalized feature matrix (n_songs × n_features)
          - feature_columns : List of feature column names that were used
    """
    df             = load_data(filepath)
    df             = clean_data(df)
    feature_df     = extract_features(df)
    feature_matrix = normalize_features(feature_df)
    feature_columns = feature_df.columns.tolist()

    return df, feature_matrix, feature_columns


# ──────────────────────────────────────────────
# Quick sanity check when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib

    data_path = pathlib.Path(__file__).parent.parent / "data" / "songs.csv"
    df, matrix, cols = preprocess(str(data_path))

    print(f"✅ Loaded {len(df)} songs")
    print(f"   Features used : {cols}")
    print(f"   Feature matrix: {matrix.shape}")
    print(df[["name", "artist"]].head())
