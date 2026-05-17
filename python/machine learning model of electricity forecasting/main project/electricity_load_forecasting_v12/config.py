"""Configuration constants for the v12 forecasting pipeline."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

CSV_PATH = "/kaggle/input/power-demand-dataset/powerdemand_5min_2021_to_2024_with weather.csv"
LOCAL_CSV_PATH = PROJECT_ROOT / "datasets" / "powerdemand_5min_2021_to_2024_with weather.csv"

FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday",
    "weekend",
    "is_peak_hour",
    "is_day",
    "temp",
    "dwpt",
    "rhum",
    "wspd",
    "pres",
    "wdir_sin",
    "wdir_cos",
    "temp_hour",
    "temp_x_peak",
    "lag_12",
    "lag_288",
    "lag_2016",
    "roll_mean_12",
    "roll_std_12",
    "roll_max_12",
    "roll_min_12",
]

TARGET = "load"
SPLIT_DATE = "2024-01-01"
RANDOM_STATE = 42

OPTUNA_TRIALS = 50
OPTUNA_CV = 3
LGBM_TRIALS = 50
LGBM_CV = 3

LOOKBACK = 24
BATCH_SIZE = 256
MAX_EPOCHS = 50

PLOT_PERIODS = 7 * 288
ZOOM = 2 * 288

COLORS = {
    "Naive Baseline": "#95A5A6",
    "Linear Regression": "#E74C3C",
    "Decision Tree": "#F39C12",
    "Random Forest": "#27AE60",
    "KNN": "#8E44AD",
    "Gradient Boosting": "#2980B9",
    "XGBoost": "#E67E22",
    "XGBoost (Tuned)": "#C0392B",
    "LightGBM": "#1ABC9C",
    "LightGBM (Tuned)": "#148F77",
    "Ensemble (XGB+LGBM)": "#6C3483",
    "LSTM": "#D4145A",
}

