# =============================================================================
# A Comparative Study of LSTM, GRU, XGBoost, and Random Forest Models
# for Electricity Load Forecasting Across Diverse Power Systems
# =============================================================================
# Kaggle-ready, reproducible, modular Python codebase
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: CONFIGURATION (edit these before running)
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

# ── File paths ────────────────────────────────────────────────────────────────
DELHI_PATH = "/kaggle/input/delhi-electricity-demand/delhi_electricity_data.csv"
UK_PATH    = "/kaggle/input/uk-national-grid/uk_historic_demand.csv"
OUTPUT_DIR = "/kaggle/working/outputs"

# ── Column mappings ───────────────────────────────────────────────────────────
DELHI_CONFIG = {
    "datetime_col": "datetime",       # actual column in Delhi CSV
    "target_col":   "Power demand",   # actual column in Delhi CSV
    "freq_minutes": 5,
    "dataset_name": "Delhi",
    # Optional weather columns to include as extra features (set [] to disable)
    "extra_features": ["temp", "rhum", "wspd", "pres"],
}

UK_CONFIG = {
    "datetime_col": "settlement_date",  # actual column in UK CSV (lowercase)
    "target_col":   "nd",               # actual column in UK CSV (lowercase)
    "freq_minutes": 30,
    "dataset_name": "UK National Grid",
    "extra_features": ["is_holiday"],   # available extra feature in UK dataset
}

# ── Lookback windows (in time-steps) ─────────────────────────────────────────
DELHI_LOOKBACK = 288   # 288 × 5 min = 24 h
UK_LOOKBACK    = 48    # 48 × 30 min = 24 h

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # remainder

# ── Lag & rolling window sizes (in time-steps per dataset) ───────────────────
DELHI_LAGS           = [1, 12, 288, 576]   # 5min, 1h, 24h, 48h
DELHI_ROLLING_WINDOW = 288                  # 24 h

UK_LAGS           = [1, 2, 48, 96]        # 30min, 1h, 24h, 48h
UK_ROLLING_WINDOW = 48                     # 24 h

# ── Training ──────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
DL_EPOCHS      = 50
DL_BATCH_SIZE  = 64
DL_PATIENCE    = 8          # early stopping patience
XGB_N_ROUNDS   = 500
XGB_PATIENCE   = 20
RF_N_ESTIMATORS = 200

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = {
    "Random Forest": "#2196F3",
    "XGBoost":       "#FF9800",
    "LSTM":          "#9C27B0",
    "GRU":           "#4CAF50",
    "Ensemble":      "#F44336",
    "Deep Learning": "#00BCD4",
}

# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================

def load_dataset(path, cfg):
    """Load a CSV, parse datetime, sort, drop duplicates, keep extra features."""
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names to strip accidental whitespace
    df.columns = df.columns.str.strip()

    dt_col     = cfg["datetime_col"]
    target_col = cfg["target_col"]

    # Safety: abort early with a clear message if columns are missing
    missing = [c for c in [dt_col, target_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{cfg['dataset_name']}] Column(s) not found: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df[dt_col] = pd.to_datetime(df[dt_col], infer_datetime_format=True)
    df = df.sort_values(dt_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[dt_col]).reset_index(drop=True)

    # Build keep-list: datetime + target + any extra features present in df
    extra_cols = [c for c in cfg.get("extra_features", []) if c in df.columns]
    keep_cols  = [dt_col, target_col] + extra_cols
    df = df[keep_cols].copy()

    df = df.rename(columns={dt_col: "datetime", target_col: "load"})
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df.dropna(subset=["load"]).reset_index(drop=True)

    # Forward-fill extra feature NaNs (e.g. weather recorded less frequently)
    for c in extra_cols:
        df[c] = df[c].ffill().bfill()

    print(f"[{cfg['dataset_name']}] Loaded {len(df):,} rows | "
          f"{df['datetime'].min()} → {df['datetime'].max()} | "
          f"Extra features: {extra_cols if extra_cols else 'none'}")
    return df

# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================

def add_temporal_features(df):
    """Add hour, day-of-week, month, quarter, weekend, and cyclical encodings.
    Safe to call even if some columns already exist (e.g. Delhi has hour/month)."""
    dt = df["datetime"]
    # Always recompute from datetime to avoid any pre-existing column conflicts
    df["hour"]       = dt.dt.hour
    df["dayofweek"]  = dt.dt.dayofweek
    df["month"]      = dt.dt.month
    df["quarter"]    = dt.dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_rolling_features(df, lags, rolling_window):
    """Add lag and rolling statistics on the load column."""
    for lag in lags:
        df[f"lag_{lag}"] = df["load"].shift(lag)

    df[f"roll_mean_{rolling_window}"] = df["load"].shift(1).rolling(rolling_window).mean()
    df[f"roll_std_{rolling_window}"]  = df["load"].shift(1).rolling(rolling_window).std()
    df[f"roll_min_{rolling_window}"]  = df["load"].shift(1).rolling(rolling_window).min()
    df[f"roll_max_{rolling_window}"]  = df["load"].shift(1).rolling(rolling_window).max()
    return df


# Columns that may exist in raw Delhi CSV that we recompute ourselves
_RECOMPUTED_COLS = ["hour", "month", "day", "year", "minute", "moving_avg_3"]

def engineer_features(df, lags, rolling_window):
    # Drop any pre-existing columns we will recompute to avoid duplicates/conflicts
    cols_to_drop = [c for c in _RECOMPUTED_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    df = add_temporal_features(df)
    df = add_lag_rolling_features(df, lags, rolling_window)
    df = df.dropna().reset_index(drop=True)
    return df

# =============================================================================
# SECTION 4: TRAIN / VAL / TEST SPLIT
# =============================================================================

def chronological_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train = df.iloc[:t1].copy()
    val   = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test

# =============================================================================
# SECTION 5: SEQUENCE CREATION FOR LSTM / GRU
# =============================================================================

def make_sequences(series, lookback):
    """Convert a 1-D array to (X, y) sequences with shape (N, lookback, 1)."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X)[..., np.newaxis], np.array(y)


def prepare_dl_data(train, val, test, lookback):
    """Scale load column and build LSTM/GRU sequences."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[["load"]]).flatten()
    val_scaled   = scaler.transform(val[["load"]]).flatten()
    test_scaled  = scaler.transform(test[["load"]]).flatten()

    # Prepend tail of train to val and test so sequences don't lose initial rows
    val_input  = np.concatenate([train_scaled[-lookback:], val_scaled])
    test_input = np.concatenate([val_scaled[-lookback:], test_scaled])

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_val,   y_val   = make_sequences(val_input, lookback)
    X_test,  y_test  = make_sequences(test_input, lookback)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# =============================================================================
# SECTION 6: TABULAR FEATURE MATRIX (Random Forest / XGBoost)
# =============================================================================

FEATURE_COLS_BASE = [
    "hour", "dayofweek", "month", "quarter", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Extra dataset-specific columns that may be present after load_dataset
DELHI_EXTRA_FEATURES = ["temp", "rhum", "wspd", "pres"]
UK_EXTRA_FEATURES    = ["is_holiday"]

def get_feature_cols(df):
    lag_cols     = [c for c in df.columns if c.startswith("lag_")]
    rolling_cols = [c for c in df.columns if c.startswith("roll_")]
    # Include any extra features that survived into this df
    extra = [c for c in (DELHI_EXTRA_FEATURES + UK_EXTRA_FEATURES)
             if c in df.columns]
    return FEATURE_COLS_BASE + extra + lag_cols + rolling_cols


def prepare_tabular(train, val, test):
    feat_cols = get_feature_cols(train)
    X_train, y_train = train[feat_cols].values, train["load"].values
    X_val,   y_val   = val[feat_cols].values,   val["load"].values
    X_test,  y_test  = test[feat_cols].values,  test["load"].values
    return X_train, y_train, X_val, y_val, X_test, y_test

# =============================================================================
# SECTION 7: METRICS
# =============================================================================

def mape_score(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mape_score(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# =============================================================================
# SECTION 8: RANDOM FOREST
# =============================================================================

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("  Training Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_test = rf.predict(X_test)
    pred_val  = rf.predict(X_val)
    metrics   = compute_metrics(y_test, pred_test)
    return rf, pred_test, pred_val, metrics

# =============================================================================
# SECTION 9: XGBOOST
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    print("  Training XGBoost …")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "rmse",
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "seed":             RANDOM_SEED,
        "verbosity":        0,
    }

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=XGB_N_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=XGB_PATIENCE,
        evals_result=evals_result,
        verbose_eval=False,
    )

    pred_test = model.predict(dtest)
    pred_val  = model.predict(dval)
    metrics   = compute_metrics(y_test, pred_test)

    train_rmse = evals_result["train"]["rmse"]
    val_rmse   = evals_result["val"]["rmse"]
    return model, pred_test, pred_val, metrics, train_rmse, val_rmse

# =============================================================================
# SECTION 10: BUILD LSTM / GRU ARCHITECTURE
# =============================================================================

def build_rnn_model(model_type, lookback, units=64, dropout=0.2):
    """Build either an LSTM or GRU model."""
    assert model_type in ("LSTM", "GRU")
    model = Sequential(name=model_type)
    RNNLayer = LSTM if model_type == "LSTM" else GRU

    model.add(RNNLayer(units, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(dropout))
    model.add(RNNLayer(units // 2))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model


def train_rnn(model_type, X_train, y_train, X_val, y_val, X_test, y_test, scaler, lookback):
    print(f"  Training {model_type} …")
    model = build_rnn_model(model_type, lookback)

    es = EarlyStopping(
        monitor="val_loss",
        patience=DL_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred_test   = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = compute_metrics(y_test_orig, pred_test)
    return model, pred_test, y_test_orig, metrics, history

# =============================================================================
# SECTION 11: RUN FULL PIPELINE FOR ONE DATASET
# =============================================================================

def run_pipeline(df_raw, cfg, lags, rolling_window, lookback):
    """End-to-end pipeline; returns all results needed for figures & tables."""
    dataset_name = cfg["dataset_name"]
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    # ── Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df_raw.copy(), lags, rolling_window)

    # ── Chronological split ────────────────────────────────────────────────
    train, val, test = chronological_split(df)

    # ── Tabular data for RF / XGB ──────────────────────────────────────────
    X_tr, y_tr, X_v, y_v, X_te, y_te = prepare_tabular(train, val, test)

    # ── Sequence data for LSTM / GRU ──────────────────────────────────────
    (X_tr_dl, y_tr_dl,
     X_v_dl,  y_v_dl,
     X_te_dl, y_te_dl,
     scaler)  = prepare_dl_data(train, val, test, lookback)

    results   = {}   # model_name -> {"metrics": …, "pred": …, "actual": …}
    histories = {}   # model_name -> Keras history or XGB curves

    # ── Random Forest ──────────────────────────────────────────────────────
    rf, pred_rf, _, m_rf = train_random_forest(X_tr, y_tr, X_v, y_v, X_te, y_te)
    results["Random Forest"] = {"metrics": m_rf, "pred": pred_rf, "actual": y_te}
    print(f"  RF  → RMSE: {m_rf['RMSE']:.2f}  MAE: {m_rf['MAE']:.2f}  "
          f"MAPE: {m_rf['MAPE']:.2f}%  R²: {m_rf['R2']:.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model, pred_xgb, _, m_xgb, tr_rmse, vl_rmse = train_xgboost(
        X_tr, y_tr, X_v, y_v, X_te, y_te)
    results["XGBoost"] = {"metrics": m_xgb, "pred": pred_xgb, "actual": y_te}
    histories["XGBoost"] = {"train_rmse": tr_rmse, "val_rmse": vl_rmse}
    print(f"  XGB → RMSE: {m_xgb['RMSE']:.2f}  MAE: {m_xgb['MAE']:.2f}  "
          f"MAPE: {m_xgb['MAPE']:.2f}%  R²: {m_xgb['R2']:.4f}")

    # ── LSTM ───────────────────────────────────────────────────────────────
    lstm_model, pred_lstm, actual_lstm, m_lstm, hist_lstm = train_rnn(
        "LSTM", X_tr_dl, y_tr_dl, X_v_dl, y_v_dl, X_te_dl, y_te_dl, scaler, lookback)
    results["LSTM"] = {"metrics": m_lstm, "pred": pred_lstm, "actual": actual_lstm}
    histories["LSTM"] = hist_lstm
    print(f"  LSTM→ RMSE: {m_lstm['RMSE']:.2f}  MAE: {m_lstm['MAE']:.2f}  "
          f"MAPE: {m_lstm['MAPE']:.2f}%  R²: {m_lstm['R2']:.4f}")

    # ── GRU ────────────────────────────────────────────────────────────────
    gru_model, pred_gru, actual_gru, m_gru, hist_gru = train_rnn(
        "GRU", X_tr_dl, y_tr_dl, X_v_dl, y_v_dl, X_te_dl, y_te_dl, scaler, lookback)
    results["GRU"] = {"metrics": m_gru, "pred": pred_gru, "actual": actual_gru}
    histories["GRU"] = hist_gru
    print(f"  GRU → RMSE: {m_gru['RMSE']:.2f}  MAE: {m_gru['MAE']:.2f}  "
          f"MAPE: {m_gru['MAPE']:.2f}%  R²: {m_gru['R2']:.4f}")

    return results, histories, test

# =============================================================================
# SECTION 12: SAVE METRICS & PREDICTIONS
# =============================================================================

def save_metrics_csv(results_delhi, results_uk):
    rows = []
    for ds_name, results in [("Delhi", results_delhi), ("UK", results_uk)]:
        for model, data in results.items():
            row = {"Dataset": ds_name, "Model": model}
            row.update(data["metrics"])
            rows.append(row)
    df_metrics = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df_metrics.to_csv(path, index=False)
    print(f"\nMetrics saved → {path}")
    return df_metrics


def save_predictions_csv(results_delhi, results_uk, test_delhi, test_uk):
    frames = []
    for ds_name, results, test_df in [
        ("Delhi", results_delhi, test_delhi),
        ("UK",    results_uk,    test_uk),
    ]:
        n = min(len(r["pred"]) for r in results.values())
        df_p = pd.DataFrame({
            "Dataset": ds_name,
            "Actual":  list(results.values())[0]["actual"][:n],
        })
        for model, data in results.items():
            df_p[f"Pred_{model}"] = data["pred"][:n]
        frames.append(df_p)
    df_all = pd.concat(frames, ignore_index=True)
    path   = os.path.join(OUTPUT_DIR, "predictions.csv")
    df_all.to_csv(path, index=False)
    print(f"Predictions saved → {path}")

# =============================================================================
# SECTION 13: PRINT TABLES
# =============================================================================

def print_table(results, dataset_name):
    print(f"\n── {dataset_name} Results ─────────────────────────────────────")
    rows = []
    for model, data in results.items():
        m = data["metrics"]
        rows.append([model, f"{m['MAE']:.2f}", f"{m['RMSE']:.2f}",
                     f"{m['MAPE']:.2f}", f"{m['R2']:.4f}"])
    df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE (%)", "R²"])
    print(df.to_string(index=False))


def print_ranking_table(results_delhi, results_uk):
    print("\n── Model Ranking (by average RMSE across both datasets) ────────")
    models = list(results_delhi.keys())
    rows   = []
    for m in models:
        avg_rmse = (results_delhi[m]["metrics"]["RMSE"] +
                    results_uk[m]["metrics"]["RMSE"]) / 2
        avg_mae  = (results_delhi[m]["metrics"]["MAE"] +
                    results_uk[m]["metrics"]["MAE"]) / 2
        rows.append((m, avg_rmse, avg_mae))
    rows.sort(key=lambda x: x[1])
    df = pd.DataFrame(rows, columns=["Model", "Avg RMSE", "Avg MAE"])
    df["Rank"] = range(1, len(df) + 1)
    print(df[["Rank", "Model", "Avg RMSE", "Avg MAE"]].to_string(index=False))

# =============================================================================
# SECTION 14: FIGURE 1 – DATASET COMPARISON
# =============================================================================

def plot_fig1_dataset_comparison(df_delhi, df_uk, cfg_delhi, cfg_uk):
    """Figure 1: Representative demand profiles for both datasets."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Delhi – pick first 3 days
    n_delhi = 288 * 3   # 3 days at 5-min
    sub_d   = df_delhi.head(n_delhi)
    axes[0].plot(sub_d["datetime"], sub_d["load"], color="#E53935", lw=0.8)
    axes[0].set_title(f"(a) {cfg_delhi['dataset_name']} Electricity Demand Profile "
                       f"(3-Day Sample, {cfg_delhi['freq_minutes']}-min resolution)")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_xlabel("Date / Time")

    # UK – pick first 7 days
    n_uk  = 48 * 7    # 7 days at 30-min
    sub_u = df_uk.head(n_uk)
    axes[1].plot(sub_u["datetime"], sub_u["load"], color="#1565C0", lw=0.8)
    axes[1].set_title(f"(b) {cfg_uk['dataset_name']} Demand Profile "
                       f"(7-Day Sample, {cfg_uk['freq_minutes']}-min resolution)")
    axes[1].set_ylabel("National Demand (MW)")
    axes[1].set_xlabel("Date / Time")

    fig.suptitle("Figure 1: Dataset Demand Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_dataset_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# =============================================================================
# SECTION 15: FIGURES 2 & 3 – MODEL PERFORMANCE BAR CHARTS
# =============================================================================

def plot_performance_bars(results, dataset_name, fig_num):
    """Figures 2 & 3: Bar charts for MAE, RMSE, MAPE, R²."""
    models  = list(results.keys())
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    colors  = [PALETTE[m] for m in models]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, metric in zip(axes, metrics):
        values = [results[m]["metrics"][metric] for m in models]
        bars   = ax.bar(models, values, color=colors, edgecolor="white", width=0.55)
        ax.set_title(metric if metric != "R2" else "R²")
        ax.set_xticklabels(models, rotation=30, ha="right")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Figure {fig_num}: Model Performance on {dataset_name} Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    tag  = dataset_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"fig{fig_num}_performance_{tag}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# =============================================================================
# SECTION 16: FIGURE 4 – ENSEMBLE VS DEEP LEARNING
# =============================================================================

def plot_fig4_ensemble_vs_dl(results_delhi, results_uk):
    """Figure 4: 4-subplot grouped bar chart (one subplot per metric).
    Each metric gets its own Y-axis so MAPE (%) and R² are clearly visible
    alongside MAE and RMSE which operate on a much larger scale.
    """
    metrics      = ["MAE", "RMSE", "MAPE", "R2"]
    metric_labels = ["MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"]

    def avg_group(results, model_names):
        return {
            m: np.mean([results[n]["metrics"][m] for n in model_names])
            for m in metrics
        }

    ensemble_models = ["Random Forest", "XGBoost"]
    dl_models       = ["LSTM", "GRU"]

    groups = {
        "Ensemble (Delhi)": avg_group(results_delhi, ensemble_models),
        "DL (Delhi)":       avg_group(results_delhi, dl_models),
        "Ensemble (UK)":    avg_group(results_uk, ensemble_models),
        "DL (UK)":          avg_group(results_uk, dl_models),
    }
    group_colors = ["#F44336", "#00BCD4", "#FF9800", "#9C27B0"]
    group_labels = list(groups.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        values = [groups[g][metric] for g in group_labels]
        bars   = ax.bar(group_labels, values, color=group_colors,
                        edgecolor="white", width=0.55)

        # Value annotations above each bar
        y_offset = max(values) * 0.02 if max(values) > 0 else 0.005
        for bar, val in zip(bars, values):
            fmt = f"{val:.3f}" if metric == "R2" else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(mlabel, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

        # R² subplot: fix y-axis from 0 to 1 for clarity
        if metric == "R2":
            ax.set_ylim(0, 1.12)

    # Shared legend using proxy patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l)
                      for c, l in zip(group_colors, group_labels)]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Figure 4: Ensemble Learning vs Deep Learning – Average Performance Comparison",
                 fontsize=12, fontweight="bold", y=1.07)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_ensemble_vs_dl.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# =============================================================================
# SECTION 17: FIGURES 5 & 6 – ACTUAL vs PREDICTED
# =============================================================================

def _best_model(results, group):
    """Return the model name with lowest RMSE from a group."""
    return min(group, key=lambda m: results[m]["metrics"]["RMSE"])


def plot_forecast_comparison(results_delhi, results_uk, fig_num, model_group, group_label):
    """Figures 5 & 6: Actual vs Predicted for best model in a group."""
    best_d = _best_model(results_delhi, model_group)
    best_u = _best_model(results_uk,   model_group)

    n_show = 200   # time-steps to display for clarity

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, ds_label, best, results in [
        (axes[0], "Delhi",        best_d, results_delhi),
        (axes[1], "UK National Grid", best_u, results_uk),
    ]:
        actual = results[best]["actual"][:n_show]
        pred   = results[best]["pred"][:n_show]
        ax.plot(actual, label="Actual",    color="#212121", lw=1.2)
        ax.plot(pred,   label=f"{best} Predicted",
                color=PALETTE[best], lw=1.0, linestyle="--")
        m = results[best]["metrics"]
        ax.set_title(f"({chr(96 + axes.tolist().index(ax) + 1)}) {ds_label} – "
                     f"Best {group_label}: {best}  |  "
                     f"RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R²={m['R2']:.4f}")
        ax.set_xlabel("Time-step (test set)")
        ax.set_ylabel("Load (MW)")
        ax.legend()

    fig.suptitle(f"Figure {fig_num}: Best {group_label} Model – Actual vs Predicted",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"fig{fig_num}_{'_'.join(group_label.lower().split())}_forecast.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# =============================================================================
# SECTION 18: FIGURES 7 & 8 – LSTM / GRU LEARNING CURVES
# =============================================================================

def plot_dl_learning_curves(histories_delhi, histories_uk, model_name, fig_num):
    """Figures 7 & 8: Training vs Validation loss for LSTM or GRU."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ds_label, histories in [
        (axes[0], "Delhi",         histories_delhi),
        (axes[1], "UK National Grid", histories_uk),
    ]:
        hist = histories[model_name].history
        ax.plot(hist["loss"],     label="Train Loss", color="#E53935")
        ax.plot(hist["val_loss"], label="Val Loss",   color="#1565C0", linestyle="--")
        ax.set_title(f"({'a' if ax is axes[0] else 'b'}) {model_name} – {ds_label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()

    fig.suptitle(f"Figure {fig_num}: {model_name} Training and Validation Loss Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"fig{fig_num}_{model_name.lower()}_learning_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# =============================================================================
# SECTION 19: FIGURE 9 – XGBOOST RMSE LEARNING CURVES
# =============================================================================

def plot_fig9_xgb_curves(histories_delhi, histories_uk):
    """Figure 9: XGBoost RMSE across boosting rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ds_label, histories in [
        (axes[0], "Delhi",         histories_delhi),
        (axes[1], "UK National Grid", histories_uk),
    ]:
        tr_rmse = histories["XGBoost"]["train_rmse"]
        vl_rmse = histories["XGBoost"]["val_rmse"]
        rounds  = range(1, len(tr_rmse) + 1)
        ax.plot(rounds, tr_rmse, label="Train RMSE", color="#E53935")
        ax.plot(rounds, vl_rmse, label="Val RMSE",   color="#1565C0", linestyle="--")
        ax.set_title(f"({'a' if ax is axes[0] else 'b'}) XGBoost RMSE – {ds_label}")
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("RMSE")
        ax.legend()

    fig.suptitle("Figure 9: XGBoost RMSE Learning Curves Across Boosting Rounds",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig9_xgb_rmse_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# =============================================================================
# SECTION 20: SUMMARY TABLE (DataFrame)
# =============================================================================

def build_summary_table(results_delhi, results_uk, df_metrics):
    print("\n" + "═" * 65)
    print("  COMBINED COMPARISON TABLE (Both Datasets)")
    print("═" * 65)
    print(df_metrics.to_string(index=False))

    print("\n  CROSS-DATASET AVERAGE PERFORMANCE:")
    for model in results_delhi.keys():
        md = results_delhi[model]["metrics"]
        mu = results_uk[model]["metrics"]
        print(f"  {model:<16} | "
              f"Avg MAE={( md['MAE']  + mu['MAE'])  / 2:.2f}  "
              f"Avg RMSE={( md['RMSE'] + mu['RMSE']) / 2:.2f}  "
              f"Avg MAPE={( md['MAPE'] + mu['MAPE']) / 2:.2f}%  "
              f"Avg R²={( md['R2']   + mu['R2'])   / 2:.4f}")

# =============================================================================
# SECTION 21: MAIN ENTRY POINT
# =============================================================================

def main():
    print("\n" + "█" * 65)
    print("  ELECTRICITY LOAD FORECASTING – COMPARATIVE STUDY")
    print("  LSTM | GRU | XGBoost | Random Forest")
    print("█" * 65)

    # ── Load raw data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading datasets …")
    df_delhi = load_dataset(DELHI_PATH, DELHI_CONFIG)
    df_uk    = load_dataset(UK_PATH,    UK_CONFIG)

    # ── Figure 1 ──────────────────────────────────────────────────────────
    print("\n[2/5] Generating Figure 1 (dataset comparison) …")
    plot_fig1_dataset_comparison(df_delhi, df_uk, DELHI_CONFIG, UK_CONFIG)

    # ── Pipelines ─────────────────────────────────────────────────────────
    print("\n[3/5] Running pipeline on Delhi dataset …")
    results_delhi, histories_delhi, test_delhi = run_pipeline(
        df_delhi, DELHI_CONFIG, DELHI_LAGS, DELHI_ROLLING_WINDOW, DELHI_LOOKBACK)

    print("\n[3/5] Running pipeline on UK dataset …")
    results_uk, histories_uk, test_uk = run_pipeline(
        df_uk, UK_CONFIG, UK_LAGS, UK_ROLLING_WINDOW, UK_LOOKBACK)

    # ── Print tables ──────────────────────────────────────────────────────
    print("\n[4/5] Generating tables …")
    print_table(results_delhi, "Delhi")
    print_table(results_uk,    "UK National Grid")
    print_ranking_table(results_delhi, results_uk)

    # ── Save CSV ──────────────────────────────────────────────────────────
    df_metrics = save_metrics_csv(results_delhi, results_uk)
    save_predictions_csv(results_delhi, results_uk, test_delhi, test_uk)
    build_summary_table(results_delhi, results_uk, df_metrics)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n[5/5] Generating all figures …")

    # Fig 2: Delhi performance bars
    plot_performance_bars(results_delhi, "Delhi", 2)

    # Fig 3: UK performance bars
    plot_performance_bars(results_uk, "UK National Grid", 3)

    # Fig 4: Ensemble vs Deep Learning
    plot_fig4_ensemble_vs_dl(results_delhi, results_uk)

    # Fig 5: Best ensemble forecast
    plot_forecast_comparison(results_delhi, results_uk,
                             fig_num=5,
                             model_group=["Random Forest", "XGBoost"],
                             group_label="Ensemble Learning")

    # Fig 6: Best DL forecast
    plot_forecast_comparison(results_delhi, results_uk,
                             fig_num=6,
                             model_group=["LSTM", "GRU"],
                             group_label="Deep Learning")

    # Fig 7: LSTM learning curves
    plot_dl_learning_curves(histories_delhi, histories_uk, "LSTM", 7)

    # Fig 8: GRU learning curves
    plot_dl_learning_curves(histories_delhi, histories_uk, "GRU", 8)

    # Fig 9: XGBoost RMSE curves
    plot_fig9_xgb_curves(histories_delhi, histories_uk)

    print("\n" + "█" * 65)
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("█" * 65)


if __name__ == "__main__":
    main()