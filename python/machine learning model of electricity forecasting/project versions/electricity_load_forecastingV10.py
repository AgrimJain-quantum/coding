# ============================================================
# ELECTRICITY LOAD FORECASTING — IMPROVED ML PIPELINE  v10
# Dataset  : Power Demand 2021–2024 | 5-minute intervals
# Target   : Power Demand (MW)
# Models   : Naive Baseline, Linear Regression, Decision Tree,
#            Random Forest, KNN, Gradient Boosting, XGBoost,
#            LightGBM, LSTM
# Tuning   : Optuna — XGBoost (n_estimators, max_depth, lr)
# Metrics  : MAE, RMSE, R², MAPE
# Author   : Academic Project — Clean & Modular Pipeline
# ============================================================

# ── SECTION 1: Imports ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor
import lightgbm              as lgb
import optuna
from optuna.samplers         import TPESampler

# Deep learning imports (LSTM)
import tensorflow as tf
from tensorflow.keras.models   import Sequential
from tensorflow.keras.layers   import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
sns.set_context("talk")

print("✅ All libraries imported successfully.\n")


# ============================================================
# SECTION 2: LOAD & CLEAN DATASET
# ============================================================
CSV_PATH = r"C:\Users\Agrim Jain\Desktop\Coding\coding\python\machine learning model of electricity forecasting\datasets\powerdemand_5min_2021_to_2024_with weather.csv"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Drop irrelevant index column and pre-computed rolling average
df.drop(columns=["Unnamed: 0", "moving_avg_3"], inplace=True)

# Rename target
df = df.rename(columns={"Power demand": "load"})

# Parse datetime and sort
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print(f"📂 Dataset loaded : {len(df):,} rows")
print(f"   Date range     : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"   Load range     : {df['load'].min():.1f} – {df['load'].max():.1f} MW\n")


# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================
print("⚙️  Engineering features …")

# ── 3a. Fix wind direction (circular 0–360°) ────────────────
# Raw wdir is meaningless as a linear number (359° ≈ 1°)
df["wdir"] = df["wdir"].ffill()
df["wdir_sin"] = np.sin(2 * np.pi * df["wdir"] / 360)
df["wdir_cos"] = np.cos(2 * np.pi * df["wdir"] / 360)
df.drop(columns=["wdir"], inplace=True)

# ── 3b. Cyclical time encoding ───────────────────────────────
# sin/cos preserves circular continuity (23:55 → 00:00)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Drop raw and redundant time columns
df.drop(columns=["hour", "month", "minute", "day", "year"], inplace=True)

# ── 3c. Binary time flags ────────────────────────────────────
df["weekday"]      = df["datetime"].dt.weekday
df["weekend"]      = (df["weekday"] >= 5).astype(int)
df["is_peak_hour"] = df["datetime"].dt.hour.between(18, 21).astype(int)
df["is_day"]       = df["datetime"].dt.hour.between(6, 18).astype(int)

# ── 3d. Interaction features ─────────────────────────────────
# V9: raw-hour proxy (kept — captures gradual intraday temp effect)
df["temp_hour"] = df["temp"] * df["datetime"].dt.hour

# V10 NEW: temp × is_peak_hour — highlights thermal load at peak demand hours
# Motivation: AC/heating load spikes when it's hot/cold AND it's peak time.
# Unlike temp_hour (continuous hour × temp), this is a binary gate that fires
# only during the 18–21 window, making the signal sharper and more interpretable.
df["temp_x_peak"] = df["temp"] * df["is_peak_hour"]

# ── 3e. Lag features (lag_1 removed — causes 99% dominance) ─
df["lag_12"]   = df["load"].shift(12)    # 1 hour ago
df["lag_288"]  = df["load"].shift(288)   # 24 hours ago
df["lag_2016"] = df["load"].shift(2016)  # 7 days ago

# ── 3f. Rolling statistics ───────────────────────────────────
df["roll_mean_12"] = df["load"].shift(1).rolling(12).mean()
df["roll_std_12"]  = df["load"].shift(1).rolling(12).std()
df["roll_max_12"]  = df["load"].shift(1).rolling(12).max()
df["roll_min_12"]  = df["load"].shift(1).rolling(12).min()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Final feature set ────────────────────────────────────────
FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday", "weekend", "is_peak_hour", "is_day",
    "temp", "dwpt", "rhum", "wspd", "pres",
    "wdir_sin", "wdir_cos",
    "temp_hour",
    "temp_x_peak",          # NEW in V10
    "lag_12", "lag_288", "lag_2016",
    "roll_mean_12", "roll_std_12", "roll_max_12", "roll_min_12",
]
TARGET = "load"

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise KeyError(f"❌ Missing features: {missing}")

print(f"✅ {len(FEATURES)} features confirmed.")
print(f"   Final dataset  : {len(df):,} rows\n")
print("   Features used  :")
for i, f in enumerate(FEATURES, 1):
    print(f"     {i:2}. {f}")
print()


# ============================================================
# SECTION 4: TRAIN / TEST SPLIT (Chronological)
# ============================================================
SPLIT_DATE = "2024-01-01"

train_mask = df["datetime"] < SPLIT_DATE
test_mask  = df["datetime"] >= SPLIT_DATE

X_train    = df.loc[train_mask, FEATURES]
y_train    = df.loc[train_mask, TARGET]
X_test     = df.loc[test_mask,  FEATURES]
y_test     = df.loc[test_mask,  TARGET]
dates_test = df.loc[test_mask,  "datetime"].reset_index(drop=True)

print(f"📊 Train : {len(X_train):,} samples  (2021-01-01 → 2023-12-31)")
print(f"   Test  : {len(X_test):,}  samples  (2024-01-01 → end)\n")

# Scale for distance/linear models only
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ============================================================
# SECTION 5: NAIVE BASELINE
# ============================================================
naive_pred = X_test["lag_288"].values
print("📏 Naive baseline: prediction = load(t-288) [same time yesterday]\n")


# ============================================================
# SECTION 6: MODEL DEFINITIONS
# ============================================================
# Format: name → (model_object, "scaled" | "raw")
# scaled = StandardScaler applied (LR, KNN)
# raw    = original features used (tree-based models)

models = {
    "Linear Regression":  (LinearRegression(),                                "scaled"),
    "Decision Tree":      (DecisionTreeRegressor(max_depth=10,
                                                  min_samples_leaf=10,
                                                  random_state=42),           "raw"),
    "Random Forest":      (RandomForestRegressor(n_estimators=150,
                                                  max_depth=15,
                                                  min_samples_leaf=4,
                                                  n_jobs=-1,
                                                  random_state=42),           "raw"),
    "KNN":                (KNeighborsRegressor(n_neighbors=10,
                                               weights="distance",
                                               n_jobs=-1),                    "scaled"),
    "Gradient Boosting":  (GradientBoostingRegressor(n_estimators=200,
                                                      max_depth=5,
                                                      learning_rate=0.05,
                                                      subsample=0.8,
                                                      random_state=42),       "raw"),
    "XGBoost":            (XGBRegressor(n_estimators=300,
                                        max_depth=6,
                                        learning_rate=0.05,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        reg_alpha=0.1,       # L1 regularisation
                                        reg_lambda=1.0,      # L2 regularisation
                                        n_jobs=-1,
                                        random_state=42,
                                        verbosity=0),                         "raw"),

    # ── V10 NEW: LightGBM ────────────────────────────────────
    # Advantages over XGBoost on large datasets:
    #   • Leaf-wise (best-first) tree growth → faster convergence
    #   • Histogram-based binning → lower memory, faster training
    #   • Native support for categorical features (not used here, but available)
    # Parameters mirror the XGBoost config for a fair initial comparison.
    "LightGBM":           (lgb.LGBMRegressor(n_estimators=300,
                                              max_depth=6,
                                              learning_rate=0.05,
                                              subsample=0.8,
                                              colsample_bytree=0.8,
                                              reg_alpha=0.1,
                                              reg_lambda=1.0,
                                              n_jobs=-1,
                                              random_state=42,
                                              verbose=-1),                    "raw"),
}


# ============================================================
# SECTION 7: TRAIN ALL MODELS + EVALUATE
# ============================================================
def compute_metrics(name, y_true, y_pred):
    """Compute MAE, RMSE, R², MAPE for a model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                           / np.array(y_true))) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}

all_metrics = []
all_preds   = {}

print("🤖 Training models …\n")

# Naive Baseline
m = compute_metrics("Naive Baseline", y_test, naive_pred)
all_metrics.append(m)
all_preds["Naive Baseline"] = naive_pred
print(f"  ✅ {'Naive Baseline':<22} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}"
      f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

# ML Models
for name, (model, data_type) in models.items():
    X_tr = X_train_sc if data_type == "scaled" else X_train
    X_te = X_test_sc  if data_type == "scaled" else X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    m = compute_metrics(name, y_test, preds)
    all_metrics.append(m)
    all_preds[name] = preds
    print(f"  ✅ {name:<22} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}"
          f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

# Results table sorted by RMSE
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print("\n" + "=" * 72)
print("  FULL EVALUATION RESULTS — sorted by RMSE ↑")
print("=" * 72)
print(results_df.to_string())
print("=" * 72)
print(f"\n  🏆 Best model: {best_model}\n")


# ============================================================
# SECTION 8: TIME SERIES CROSS-VALIDATION (XGBoost)
# ============================================================
print("🔁 TimeSeriesSplit cross-validation on XGBoost (5 folds) …")

xgb_cv  = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         n_jobs=-1, random_state=42, verbosity=0)
tscv    = TimeSeriesSplit(n_splits=5)
cv_maes = []

X_full = df[FEATURES]
y_full = df[TARGET]

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_full), 1):
    xgb_cv.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
    preds = xgb_cv.predict(X_full.iloc[te_idx])
    mae   = mean_absolute_error(y_full.iloc[te_idx], preds)
    cv_maes.append(mae)
    print(f"   Fold {fold}: MAE = {mae:.2f} MW")

print(f"\n   CV Mean MAE : {np.mean(cv_maes):.2f} MW")
print(f"   CV Std  MAE : {np.std(cv_maes):.2f} MW\n")


# ============================================================
# SECTION 8b: OPTUNA HYPERPARAMETER TUNING — XGBoost
# Searches: n_estimators, max_depth, learning_rate
# Strategy: 3-fold TimeSeriesSplit CV, minimise mean MAE
# ============================================================
print("🔍 Optuna tuning — XGBoost (50 trials × 3-fold CV) …\n")

OPTUNA_TRIALS = 50
OPTUNA_CV     = 3                    # folds kept small for speed; raise to 5 for final run

_tscv_opt = TimeSeriesSplit(n_splits=OPTUNA_CV)

def _xgb_objective(trial: optuna.Trial) -> float:
    """Objective: mean CV-MAE for a candidate XGBoost configuration."""
    params = dict(
        n_estimators  = trial.suggest_int("n_estimators",  100, 600, step=50),
        max_depth      = trial.suggest_int("max_depth",       3,  10),
        learning_rate  = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        # Fixed knobs kept from V8 — not re-tuned here
        subsample      = 0.8,
        colsample_bytree = 0.8,
        reg_alpha      = 0.1,
        reg_lambda     = 1.0,
        n_jobs         = -1,
        random_state   = 42,
        verbosity      = 0,
        early_stopping_rounds = 20,    # stop early if no improvement
        eval_metric    = "mae",
    )

    fold_maes = []
    for tr_idx, va_idx in _tscv_opt.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        mdl = XGBRegressor(**params)
        mdl.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        fold_maes.append(mean_absolute_error(y_va, mdl.predict(X_va)))

    return float(np.mean(fold_maes))


study = optuna.create_study(
    direction = "minimize",
    sampler   = TPESampler(seed=42),
    study_name = "xgb_load_forecast",
)
study.optimize(_xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_cv_mae = study.best_value

print(f"\n✅ Optuna finished — best CV-MAE: {best_cv_mae:.4f} MW")
print("   Best hyperparameters found:")
for k, v in best_params.items():
    print(f"     {k:<20} = {v}")

# ── Re-train tuned XGBoost on the full training set ─────────
xgb_tuned = XGBRegressor(
    **best_params,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    n_jobs           = -1,
    random_state     = 42,
    verbosity        = 0,
)
xgb_tuned.fit(X_train, y_train)
tuned_preds  = xgb_tuned.predict(X_test)
m_tuned      = compute_metrics("XGBoost (Tuned)", y_test, tuned_preds)

# Add tuned model to results and predictions
all_metrics.append(m_tuned)
all_preds["XGBoost (Tuned)"] = tuned_preds
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print(f"\n📊 XGBoost (default) vs XGBoost (Tuned) — Test Set 2024")
print(f"   {'Metric':<10}  {'Default':>10}  {'Tuned':>10}")
print(f"   {'-'*34}")
for metric in ["MAE", "RMSE", "R²", "MAPE (%)"]:
    default_val = results_df.loc["XGBoost", metric] if "XGBoost" in results_df.index else float("nan")
    tuned_val   = results_df.loc["XGBoost (Tuned)", metric]
    print(f"   {metric:<10}  {default_val:>10.4f}  {tuned_val:>10.4f}")

# ── Optuna convergence plot (saved to file) ──────────────────
trial_nums  = [t.number + 1                for t in study.trials]
trial_maes  = [t.value                     for t in study.trials]
running_min = pd.Series(trial_maes).cummin().tolist()

fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
ax_opt.scatter(trial_nums, trial_maes,  color="#95A5A6", s=18, alpha=0.6, label="Trial MAE")
ax_opt.plot(trial_nums, running_min,    color="#E67E22", linewidth=2,   label="Best so far")
ax_opt.set_xlabel("Trial number")
ax_opt.set_ylabel("CV-MAE (MW)")
ax_opt.set_title("Optuna Convergence — XGBoost Hyperparameter Search",
                 fontsize=12, fontweight="bold")
ax_opt.legend()
plt.tight_layout()
plt.savefig("plot_optuna_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot_optuna_convergence.png\n")

# Register tuned model colour for later plots
COLORS = globals().get("COLORS", {})
COLORS["XGBoost (Tuned)"] = "#C0392B"     # deep red — distinct from default orange


# ============================================================
# SECTION 8c: LSTM — Sequence-to-One Forecasting  (NEW V10)
# ============================================================
# Architecture overview:
#   • Input  : sliding window of LOOKBACK = 12 timesteps (1 hour)
#   • Output : single-step forecast (load at t+1)
#   • Layers : LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)
#   • Scaler : StandardScaler fitted on train (same as other models)
#   • Train  : EarlyStopping on val_loss (patience=5), max 30 epochs
#
# Why 12-step lookback?
#   Our strongest lag features are at lag_12 (1 h) and lag_288 (24 h).
#   A 12-step window gives the LSTM the same 1-hour recency window
#   without making sequences too long to train quickly.
#   For a deeper experiment, try LOOKBACK = 288 (full 24 h).
# ============================================================
print("🧠 Training LSTM …")

LOOKBACK   = 12    # timesteps per sequence (1 hour at 5-min resolution)
BATCH_SIZE = 256
MAX_EPOCHS = 30

# ── Scale features ───────────────────────────────────────────
# Re-use the same scaler fitted on X_train (already fit above).
X_train_sc_arr = X_train_sc          # numpy array, shape (N_train, n_features)
X_test_sc_arr  = X_test_sc           # numpy array, shape (N_test,  n_features)
y_train_arr    = y_train.values
y_test_arr     = y_test.values

# Also scale the target so the LSTM loss is on a normalised scale,
# then invert to MW for evaluation.
y_scaler = StandardScaler()
y_train_sc = y_scaler.fit_transform(y_train_arr.reshape(-1, 1)).ravel()
y_test_sc  = y_scaler.transform(y_test_arr.reshape(-1, 1)).ravel()


def make_sequences(X, y, lookback):
    """Convert (N, F) arrays into (N-lookback, lookback, F) sequences and (N-lookback,) targets."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])   # shape (lookback, n_features)
        ys.append(y[i])                   # scalar target
    return np.array(Xs), np.array(ys)


X_seq_train, y_seq_train = make_sequences(X_train_sc_arr, y_train_sc, LOOKBACK)
X_seq_test,  y_seq_test  = make_sequences(X_test_sc_arr,  y_test_sc,  LOOKBACK)

print(f"   LSTM train sequences : {X_seq_train.shape}")
print(f"   LSTM test  sequences : {X_seq_test.shape}")

# ── Build model ──────────────────────────────────────────────
n_features = X_seq_train.shape[2]

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1),
], name="LSTM_LoadForecast")

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_model.summary()

# ── Train ────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = lstm_model.fit(
    X_seq_train, y_seq_train,
    validation_split = 0.1,          # last 10% of train as validation
    epochs           = MAX_EPOCHS,
    batch_size       = BATCH_SIZE,
    callbacks        = [early_stop],
    verbose          = 1,
)

# ── Evaluate ─────────────────────────────────────────────────
lstm_preds_sc  = lstm_model.predict(X_seq_test, verbose=0).ravel()
lstm_preds_mw  = y_scaler.inverse_transform(lstm_preds_sc.reshape(-1, 1)).ravel()

# Note: sequences start LOOKBACK steps into the test set, so align y_test
y_test_lstm = y_test_arr[LOOKBACK:]
# Also trim dates_test to match
dates_test_lstm = dates_test.iloc[LOOKBACK:].reset_index(drop=True)

m_lstm = compute_metrics("LSTM", y_test_lstm, lstm_preds_mw)
all_metrics.append(m_lstm)
all_preds["LSTM"] = lstm_preds_mw     # length differs by LOOKBACK from other models

# Rebuild results_df including LSTM
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print(f"\n  ✅ {'LSTM':<22} MAE={m_lstm['MAE']:7.2f}  RMSE={m_lstm['RMSE']:7.2f}"
      f"  R²={m_lstm['R²']:.4f}  MAPE={m_lstm['MAPE (%)']:.2f}%")

# ── LSTM training history plot ───────────────────────────────
fig_lstm, ax_lstm = plt.subplots(figsize=(10, 4))
ax_lstm.plot(history.history["loss"],     label="Train Loss (MSE)", color="#3498DB")
ax_lstm.plot(history.history["val_loss"], label="Val Loss (MSE)",   color="#E74C3C", linestyle="--")
ax_lstm.set_xlabel("Epoch")
ax_lstm.set_ylabel("MSE (normalised)")
ax_lstm.set_title("LSTM Training History", fontsize=12, fontweight="bold")
ax_lstm.legend()
plt.tight_layout()
plt.savefig("plot_lstm_training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot_lstm_training_history.png\n")


# ============================================================
# SECTION 9: FEATURE IMPORTANCE (XGBoost + Random Forest + LightGBM)
# ============================================================
rf_model   = models["Random Forest"][0]
xgb_model  = models["XGBoost"][0]
lgbm_model = models["LightGBM"][0]

rf_imp   = pd.Series(rf_model.feature_importances_,   index=FEATURES).sort_values(ascending=False)
xgb_imp  = pd.Series(xgb_model.feature_importances_,  index=FEATURES).sort_values(ascending=False)
lgbm_imp = pd.Series(lgbm_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("🌲 Random Forest — Top 10 Feature Importances:")
for feat, imp in rf_imp.head(10).items():
    print(f"  {feat:<22}  {imp:.4f}  {'█' * int(imp * 80)}")

print("\n⚡ XGBoost — Top 10 Feature Importances:")
for feat, imp in xgb_imp.head(10).items():
    print(f"  {feat:<22}  {imp:.4f}  {'█' * int(imp * 80)}")

print("\n💡 LightGBM — Top 10 Feature Importances:")
for feat, imp in lgbm_imp.head(10).items():
    print(f"  {feat:<22}  {imp:.4f}  {'█' * int(imp * 80)}")


# ============================================================
# SECTION 10: VISUALIZATION — 6 SEABORN PLOTS (separate files)
# ============================================================
PLOT_PERIODS = 7 * 288   # last 7 days of test set

plot_dates  = dates_test.values[-PLOT_PERIODS:]
plot_actual = y_test.values[-PLOT_PERIODS:]

COLORS = {
    "Naive Baseline":    "#95A5A6",
    "Linear Regression": "#E74C3C",
    "Decision Tree":     "#F39C12",
    "Random Forest":     "#27AE60",
    "KNN":               "#8E44AD",
    "Gradient Boosting": "#2980B9",
    "XGBoost":           "#E67E22",
    "XGBoost (Tuned)":   "#C0392B",
    "LightGBM":          "#1ABC9C",   # teal — distinct from all existing colours
    # LSTM is excluded from multi-model overlays (sequence offset causes alignment issues)
}

# Global seaborn theme for all plots
sns.set_theme(style="darkgrid", palette="deep")
sns.set_context("talk", font_scale=0.85)


# ── Plot 1: Actual vs Predicted (all models, last 7 days) ────
# Build a long-form DataFrame — required for seaborn lineplot
rows = []
for t, a in zip(plot_dates, plot_actual):
    rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for name, preds in all_preds.items():
    if name == "LSTM":
        continue   # LSTM has a different length due to sequence offset
    for t, p in zip(plot_dates, preds[-PLOT_PERIODS:]):
        rows.append({"datetime": t, "Power Demand (MW)": p, "Model": name})

plot1_df = pd.DataFrame(rows)
plot1_df["datetime"] = pd.to_datetime(plot1_df["datetime"])

color_map = {"Actual": "#2C3E50", **COLORS}
dash_map  = {m: (4, 2) if m == "Naive Baseline" else "" for m in color_map}

fig, ax = plt.subplots(figsize=(20, 6))
sns.lineplot(
    data=plot1_df, x="datetime", y="Power Demand (MW)",
    hue="Model", palette=color_map,
    linewidth=0.9, ax=ax, legend="full"
)
# Make Actual line thicker and on top
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.2)
        line.set_zorder(10)
ax.set_title("Actual vs Predicted Power Demand — Last 7 Days of Test Set (2024)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot1_actual_vs_predicted.png")


# ── Plot 2: Residuals (seaborn lineplot, one panel per model) ─
# Exclude LSTM (length mismatch)
residual_models = {k: v for k, v in all_preds.items() if k != "LSTM"}
n_models = len(residual_models)
n_cols = 2
n_rows = (n_models + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4), sharex=True)
axes = axes.flatten()

for i, (name, preds) in enumerate(residual_models.items()):
    residuals = plot_actual - preds[-PLOT_PERIODS:]
    res_df    = pd.DataFrame({"datetime": pd.to_datetime(plot_dates),
                               "Residual": residuals})
    ax = axes[i]
    sns.lineplot(data=res_df, x="datetime", y="Residual",
                 color=COLORS.get(name, "#AAB7B8"), linewidth=0.75, alpha=0.9, ax=ax)
    ax.axhline(0, color="#2C3E50", linewidth=1.0, linestyle="--")
    ax.fill_between(res_df["datetime"], res_df["Residual"], 0,
                    where=(res_df["Residual"] > 0), alpha=0.15, color="#27AE60")
    ax.fill_between(res_df["datetime"], res_df["Residual"], 0,
                    where=(res_df["Residual"] < 0), alpha=0.15, color="#E74C3C")
    ax.set_title(f"{name}  |  MAE = {results_df.loc[name, 'MAE']:.1f} MW",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Residual (MW)", fontsize=8)
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Residuals (Actual − Predicted) — Last 7 Days of Test Set",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot2_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot2_residuals.png")


# ── Plot 3: Feature Importances — RF vs XGBoost vs LightGBM ──
fig, axes = plt.subplots(1, 3, figsize=(26, 8))

for ax, (title, imp, pal_color) in zip(axes, [
    ("Random Forest", rf_imp.head(15),   "Greens_r"),
    ("XGBoost",       xgb_imp.head(15),  "Oranges_r"),
    ("LightGBM",      lgbm_imp.head(15), "BuGn_r"),
]):
    imp_df = imp.reset_index()
    imp_df.columns = ["Feature", "Importance"]

    sns.barplot(
        data=imp_df, y="Feature", x="Importance",
        palette=pal_color, edgecolor="white",
        linewidth=0.6, ax=ax, orient="h"
    )
    for bar, val in zip(ax.patches, imp_df["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)

    ax.set_title(f"{title} — Top 15 Feature Importances",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")

plt.suptitle("Feature Importances: Random Forest vs XGBoost vs LightGBM",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_feature_importances.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot3_feature_importances.png")


# ── Plot 4: Model Comparison (seaborn barplot, 3 metrics) ────
metrics_to_plot = ["MAE", "RMSE", "MAPE (%)"]
compare_df   = results_df[metrics_to_plot].reset_index()
compare_long = compare_df.melt(id_vars="Model",
                                value_vars=metrics_to_plot,
                                var_name="Metric",
                                value_name="Value")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, metric in zip(axes, metrics_to_plot):
    sub = compare_long[compare_long["Metric"] == metric].copy()
    sub = sub.sort_values("Value")
    bar_colors = [COLORS.get(m, "#AAB7B8") for m in sub["Model"]]

    bars = sns.barplot(
        data=sub, x="Model", y="Value",
        palette=bar_colors, edgecolor="white",
        linewidth=0.6, ax=ax
    )
    for bar, val in zip(ax.patches, sub["Value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.patches[0].set_edgecolor("gold")
    ax.patches[0].set_linewidth(3)

    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=35)

plt.suptitle("Model Comparison — Test Set 2024  (Gold border = best per metric)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot4_model_comparison.png")


# ── Plot 5: XGBoost Zoom — last 2 days (seaborn lineplot) ────
ZOOM = 2 * 288
zoom_rows = []
for t, a in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
    zoom_rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for name in ["XGBoost", "Random Forest", "LightGBM"]:
    for t, p in zip(plot_dates[-ZOOM:], all_preds[name][-ZOOM:]):
        zoom_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": name})

zoom_df = pd.DataFrame(zoom_rows)
zoom_df["datetime"] = pd.to_datetime(zoom_df["datetime"])

zoom_palette = {"Actual":        "#2C3E50",
                "XGBoost":       "#E67E22",
                "Random Forest": "#27AE60",
                "LightGBM":      "#1ABC9C"}

fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(
    data=zoom_df, x="datetime", y="Power Demand (MW)",
    hue="Model", palette=zoom_palette,
    linewidth=1.2, ax=ax
)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.2)
        line.set_zorder(10)

ax.set_title("XGBoost vs Random Forest vs LightGBM — Zoomed View (Last 2 Days)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot5_xgboost_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot5_xgboost_zoom.png")


# ── Plot 6 (NEW V10): LSTM vs Best Tree Model — Zoomed ───────
# Use dates_test_lstm for alignment (LSTM predictions start LOOKBACK steps in)
ZOOM_LSTM = min(2 * 288, len(lstm_preds_mw))
lstm_zoom_rows = []
for t, a in zip(dates_test_lstm.values[-ZOOM_LSTM:], y_test_lstm[-ZOOM_LSTM:]):
    lstm_zoom_rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for t, p in zip(dates_test_lstm.values[-ZOOM_LSTM:], lstm_preds_mw[-ZOOM_LSTM:]):
    lstm_zoom_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": "LSTM"})
# Add best non-LSTM model for comparison
best_non_lstm = results_df[results_df.index != "LSTM"].index[0]
for t, p in zip(dates_test_lstm.values[-ZOOM_LSTM:], all_preds[best_non_lstm][-ZOOM_LSTM:]):
    lstm_zoom_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": best_non_lstm})

lstm_zoom_df = pd.DataFrame(lstm_zoom_rows)
lstm_zoom_df["datetime"] = pd.to_datetime(lstm_zoom_df["datetime"])

lstm_zoom_pal = {"Actual":       "#2C3E50",
                 "LSTM":         "#9B59B6",
                 best_non_lstm:  COLORS.get(best_non_lstm, "#E67E22")}

fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(
    data=lstm_zoom_df, x="datetime", y="Power Demand (MW)",
    hue="Model", palette=lstm_zoom_pal,
    linewidth=1.2, ax=ax
)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.2)
        line.set_zorder(10)

ax.set_title(f"LSTM vs {best_non_lstm} — Zoomed View (Last 2 Days)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot6_lstm_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot6_lstm_zoom.png")


# ============================================================
# SECTION 11: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("  ELECTRICITY LOAD FORECASTING — FINAL SUMMARY (Test: 2024)")
print("=" * 72)
print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE%':>8}")
print("  " + "-" * 62)
for mname, row in results_df.iterrows():
    marker = "  🏆" if mname == best_model else ""
    print(f"  {mname:<25} {row['MAE']:>8.2f} {row['RMSE']:>8.2f}"
          f" {row['R²']:>8.4f} {row['MAPE (%)']:>7.2f}%{marker}")
print("=" * 72)
print(f"""
📋 CHANGES FROM PREVIOUS VERSION (V9 → V10):
  ✅ LightGBM added to model suite (Section 6, leaf-wise GBDT)
  ✅ temp_x_peak interaction feature added (temp × is_peak_hour, Section 3d)
  ✅ LSTM model added: 2-layer stacked, 12-step lookback, EarlyStopping (Section 8c)
  ✅ LSTM training history plot added (plot_lstm_training_history.png)
  ✅ Feature importance plot extended to 3 panels (RF / XGBoost / LightGBM)
  ✅ Zoom plot 5 now includes LightGBM
  ✅ New Plot 6: LSTM vs best tree model — zoomed 2-day view

🔬 NEXT STEPS:
  1. ✅ Tune XGBoost with Optuna (n_estimators, max_depth, lr) — DONE (V9)
  2. ✅ Try LightGBM — faster training on large datasets — DONE (V10)
  3. ✅ Add temp × is_peak_hour interaction feature — DONE (V10)
  4. ✅ LSTM baseline added — DONE (V10)
  5. Try Transformer (temporal self-attention) or TFT (Temporal Fusion Transformer)
  6. Tune LightGBM with Optuna (num_leaves, min_child_samples)
  7. Extend LSTM lookback to 288 steps (full 24-hour window)
  8. Ensemble: blend XGBoost (Tuned) + LightGBM predictions
""")