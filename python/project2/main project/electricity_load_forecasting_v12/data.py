"""Data loading, feature engineering, and chronological splitting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import CSV_PATH, FEATURES, LOCAL_CSV_PATH, SPLIT_DATE, TARGET


@dataclass
class PreparedData:
    df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    dates_test: pd.Series
    X_train_sc: np.ndarray
    X_test_sc: np.ndarray
    scaler: StandardScaler


def resolve_csv_path(csv_path: str | Path = CSV_PATH) -> Path | str:
    """Use Kaggle path when available, otherwise fall back to the local dataset."""
    candidate = Path(csv_path)
    if candidate.exists():
        return candidate
    if str(csv_path) == CSV_PATH and LOCAL_CSV_PATH.exists():
        return LOCAL_CSV_PATH
    return csv_path


def load_and_clean_data(csv_path: str | Path = CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the original cleanup steps."""
    resolved_path = resolve_csv_path(csv_path)

    df = pd.read_csv(resolved_path)
    df.columns = df.columns.str.strip()

    df.drop(columns=["Unnamed: 0", "moving_avg_3"], inplace=True)
    df = df.rename(columns={"Power demand": TARGET})

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Dataset loaded : {len(df):,} rows")
    print(f"   Source      : {resolved_path}")
    print(f"   Date range  : {df['datetime'].min().date()} -> {df['datetime'].max().date()}")
    print(f"   Load range  : {df[TARGET].min():.1f} - {df[TARGET].max():.1f} MW\n")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered feature set used by every model."""
    df = df.copy()
    print("Engineering features ...")

    df["wdir"] = df["wdir"].ffill()
    df["wdir_sin"] = np.sin(2 * np.pi * df["wdir"] / 360)
    df["wdir_cos"] = np.cos(2 * np.pi * df["wdir"] / 360)
    df.drop(columns=["wdir"], inplace=True)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df.drop(columns=["hour", "month", "minute", "day", "year"], inplace=True)

    df["weekday"] = df["datetime"].dt.weekday
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_peak_hour"] = df["datetime"].dt.hour.between(18, 21).astype(int)
    df["is_day"] = df["datetime"].dt.hour.between(6, 18).astype(int)

    df["temp_hour"] = df["temp"] * df["datetime"].dt.hour
    df["temp_x_peak"] = df["temp"] * df["is_peak_hour"]

    df["lag_12"] = df[TARGET].shift(12)
    df["lag_288"] = df[TARGET].shift(288)
    df["lag_2016"] = df[TARGET].shift(2016)

    shifted_load = df[TARGET].shift(1)
    df["roll_mean_12"] = shifted_load.rolling(12).mean()
    df["roll_std_12"] = shifted_load.rolling(12).std()
    df["roll_max_12"] = shifted_load.rolling(12).max()
    df["roll_min_12"] = shifted_load.rolling(12).min()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    print(f"{len(FEATURES)} features confirmed.")
    print(f"   Final dataset : {len(df):,} rows\n")
    print("   Features used :")
    for index, feature in enumerate(FEATURES, 1):
        print(f"     {index:2}. {feature}")
    print()

    return df


def split_and_scale(df: pd.DataFrame) -> PreparedData:
    """Create the chronological train/test split and scaled feature arrays."""
    train_mask = df["datetime"] < SPLIT_DATE
    test_mask = df["datetime"] >= SPLIT_DATE

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]
    dates_test = df.loc[test_mask, "datetime"].reset_index(drop=True)

    print(f"Train : {len(X_train):,} samples  (2021-01-01 -> 2023-12-31)")
    print(f"Test  : {len(X_test):,} samples  (2024-01-01 -> end)\n")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return PreparedData(
        df=df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dates_test=dates_test,
        X_train_sc=X_train_sc,
        X_test_sc=X_test_sc,
        scaler=scaler,
    )

