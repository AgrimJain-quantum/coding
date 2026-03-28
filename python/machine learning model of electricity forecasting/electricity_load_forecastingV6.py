# ============================================================
# ELECTRICITY LOAD FORECASTING — IMPROVED ML PIPELINE
# Dataset  : Power Demand 2021–2024 | 5-minute intervals
# Target   : Power Demand (MW)
# Models   : Naive Baseline, Linear Regression, Decision Tree,
#            Random Forest, KNN, Gradient Boosting
# Metrics  : MAE, RMSE, R², MAPE
# Author   : Academic Project — Clean & Modular Pipeline
# ============================================================

# ── SECTION 1: Imports ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
print("✅ All libraries imported successfully.\n")


# ============================================================
# SECTION 2: LOAD & CLEAN DATASET
# ============================================================
CSV_PATH = r""

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

# ── 3a. Fix wind direction (circular, 0–360°) ───────────────
# Raw wdir is meaningless as a linear number (359° ≈ 1°)
# Convert to sin/cos components to capture circular nature
df["wdir"] = df["wdir"].ffill()
df["wdir_sin"] = np.sin(2 * np.pi * df["wdir"] / 360)
df["wdir_cos"] = np.cos(2 * np.pi * df["wdir"] / 360)
df.drop(columns=["wdir"], inplace=True)

# ── 3b. Cyclical time encoding ───────────────────────────────
# Use ONLY sin/cos — removes raw hour & month (redundant)
# sin/cos preserves circular nature (23:55 → 00:00 continuity)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Drop raw hour, month, and other redundant time columns
# minute → too granular  |  day/year/quarter → not useful for short-term
df.drop(columns=["hour", "month", "minute", "day", "year"], inplace=True)

# ── 3c. Meaningful binary time flags ────────────────────────
df["weekday"]      = df["datetime"].dt.weekday
df["weekend"]      = (df["weekday"] >= 5).astype(int)
df["is_peak_hour"] = df["datetime"].dt.hour.between(18, 21).astype(int)  # evening peak
df["is_day"]       = df["datetime"].dt.hour.between(6, 18).astype(int)   # daylight hours

# ── 3d. Interaction feature ──────────────────────────────────
# Temperature × hour_sin captures heat-driven demand at peak times
df["temp_hour"] = df["temp"] * df["datetime"].dt.hour

# ── 3e. Lag features (NO lag_1 — causes 99% importance dominance) ──
# lag_12   = 1 hour ago       (captures recent trend)
# lag_288  = 24 hours ago     (captures daily pattern — same time yesterday)
# lag_2016 = 7 days ago       (captures weekly seasonality)
df["lag_12"]   = df["load"].shift(12)
df["lag_288"]  = df["load"].shift(288)
df["lag_2016"] = df["load"].shift(2016)

# ── 3f. Rolling statistics (based on 1-hour window, offset by 1) ──
df["roll_mean_12"] = df["load"].shift(1).rolling(12).mean()  # 1-hr rolling mean
df["roll_std_12"]  = df["load"].shift(1).rolling(12).std()   # 1-hr demand volatility
df["roll_max_12"]  = df["load"].shift(1).rolling(12).max()   # 1-hr rolling peak
df["roll_min_12"]  = df["load"].shift(1).rolling(12).min()   # 1-hr rolling trough

# Drop rows with NaN from lag/rolling creation
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Final feature set (18 features — lean and meaningful) ───
FEATURES = [
    # Cyclical time (no raw hour/month)
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Binary time flags
    "weekday", "weekend", "is_peak_hour", "is_day",
    # Weather (wdir converted to sin/cos)
    "temp", "dwpt", "rhum", "wspd", "pres",
    "wdir_sin", "wdir_cos",
    # Interaction
    "temp_hour",
    # Lag features (no lag_1)
    "lag_12", "lag_288", "lag_2016",
    # Rolling features
    "roll_mean_12", "roll_std_12", "roll_max_12", "roll_min_12",
]
TARGET = "load"

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise KeyError(f"❌ Missing features: {missing}")

print(f"✅ {len(FEATURES)} features confirmed (lag_1 removed, wdir fixed).")
print(f"   Final dataset  : {len(df):,} rows\n")
print("   Features used  :")
for i, f in enumerate(FEATURES, 1):
    print(f"     {i:2}. {f}")
print()


# ============================================================
# SECTION 4: TRAIN / TEST SPLIT (Chronological)
# ============================================================
# Train: 2021–2023  |  Test: 2024
# Strict chronological split — no data leakage
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

# Scale features for distance/linear models
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ============================================================
# SECTION 5: NAIVE BASELINE MODEL
# ============================================================
# Prediction = load from same time yesterday (lag_288)
# This is the simplest possible forecast — all models must beat this
naive_pred = X_test["lag_288"].values
print("📏 Naive baseline defined: prediction = load(t-288) [same time yesterday]")


# ============================================================
# SECTION 6: MODEL DEFINITIONS
# ============================================================
models = {
    "Linear Regression":    (LinearRegression(),                              "scaled"),
    "Decision Tree":        (DecisionTreeRegressor(max_depth=10,
                                                   min_samples_leaf=10,
                                                   random_state=42),          "raw"),
    "Random Forest":        (RandomForestRegressor(n_estimators=150,
                                                   max_depth=15,
                                                   min_samples_leaf=4,
                                                   n_jobs=-1,
                                                   random_state=42),          "raw"),
    "KNN":                  (KNeighborsRegressor(n_neighbors=10,
                                                 weights="distance",
                                                 n_jobs=-1),                  "scaled"),
    "Gradient Boosting":    (GradientBoostingRegressor(n_estimators=200,
                                                       max_depth=5,
                                                       learning_rate=0.05,
                                                       subsample=0.8,
                                                       random_state=42),      "raw"),
}


# ============================================================
# SECTION 7: TRAIN ALL MODELS + EVALUATE
# ============================================================
def compute_metrics(name, y_true, y_pred):
    """Return a dict of evaluation metrics for a model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                           / np.array(y_true))) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}

all_metrics  = []
all_preds    = {}

print("🤖 Training models …\n")

# ── Naive Baseline ──
m = compute_metrics("Naive Baseline (lag_288)", y_test, naive_pred)
all_metrics.append(m)
all_preds["Naive Baseline (lag_288)"] = naive_pred
print(f"  ✅ Naive Baseline       — MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}"
      f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

# ── ML Models ──
for name, (model, data_type) in models.items():
    X_tr = X_train_sc if data_type == "scaled" else X_train
    X_te = X_test_sc  if data_type == "scaled" else X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    m = compute_metrics(name, y_test, preds)
    all_metrics.append(m)
    all_preds[name] = preds
    print(f"  ✅ {name:<22} — MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}"
          f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

# Results table
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")

print("\n" + "=" * 70)
print("  FULL EVALUATION RESULTS (sorted by RMSE ↑)")
print("=" * 70)
print(results_df.to_string())
print("=" * 70)
best_model = results_df.index[0]
print(f"\n  🏆 Best model: {best_model}\n")


# ============================================================
# SECTION 8: TIME SERIES CROSS-VALIDATION (BONUS)
# ============================================================
print("🔁 Running TimeSeriesSplit cross-validation on Random Forest …")

rf_cv    = RandomForestRegressor(n_estimators=100, max_depth=12,
                                  min_samples_leaf=5, n_jobs=-1, random_state=42)
tscv     = TimeSeriesSplit(n_splits=5)
cv_maes  = []

X_full = df[FEATURES]
y_full = df[TARGET]

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_full), 1):
    rf_cv.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
    preds = rf_cv.predict(X_full.iloc[te_idx])
    mae   = mean_absolute_error(y_full.iloc[te_idx], preds)
    cv_maes.append(mae)
    print(f"   Fold {fold}: MAE = {mae:.2f} MW")

print(f"\n   CV Mean MAE : {np.mean(cv_maes):.2f} MW")
print(f"   CV Std  MAE : {np.std(cv_maes):.2f} MW\n")


# ============================================================
# SECTION 9: FEATURE IMPORTANCE (Random Forest)
# ============================================================
rf_model      = models["Random Forest"][0]   # already trained
importances   = pd.Series(rf_model.feature_importances_, index=FEATURES)
imp_sorted    = importances.sort_values(ascending=False)

print("🌲 Random Forest — Feature Importances:")
for feat, imp in imp_sorted.items():
    bar = "█" * int(imp * 80)
    print(f"  {feat:<20}  {imp:.4f}  {bar}")


# ============================================================
# SECTION 10: VISUALIZATION — 4 SEPARATE PLOTS
# ============================================================
# Use last 7 days of test set for time-series plots
PLOT_PERIODS = 7 * 288      # 7 days × 288 periods/day

plot_dates  = dates_test.values[-PLOT_PERIODS:]
plot_actual = y_test.values[-PLOT_PERIODS:]

# Colors per model
COLORS = {
    "Naive Baseline (lag_288)": "#95A5A6",
    "Linear Regression":        "#E74C3C",
    "Decision Tree":            "#F39C12",
    "Random Forest":            "#27AE60",
    "KNN":                      "#8E44AD",
    "Gradient Boosting":        "#2980B9",
}

# ── PLOT 1: Actual vs Predicted (all models, last 7 days) ────
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(plot_dates, plot_actual, label="Actual", color="#2C3E50", lw=2, zorder=5)
for name, preds in all_preds.items():
    ax.plot(plot_dates, preds[-PLOT_PERIODS:],
            label=name, color=COLORS[name], lw=0.9,
            linestyle="--" if "Naive" in name else "-", alpha=0.85)
ax.set_title("Actual vs Predicted Power Demand — Last 7 Days (2024)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot1_actual_vs_predicted.png")

# ── PLOT 2: Residual plots (one panel per model) ─────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
axes = axes.flatten()
for i, (name, preds) in enumerate(all_preds.items()):
    residuals = plot_actual - preds[-PLOT_PERIODS:]
    ax = axes[i]
    ax.plot(plot_dates, residuals, color=COLORS[name], lw=0.7, alpha=0.85)
    ax.axhline(0, color="black", lw=0.9, linestyle="--")
    ax.fill_between(plot_dates, residuals, 0,
                    where=(residuals > 0), alpha=0.12, color="green")
    ax.fill_between(plot_dates, residuals, 0,
                    where=(residuals < 0), alpha=0.12, color="red")
    ax.set_title(f"{name}  |  MAE={results_df.loc[name,'MAE']:.1f} MW", fontsize=10)
    ax.set_ylabel("Residual (MW)", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.25)
plt.suptitle("Residuals (Actual − Predicted) — Last 7 Days", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot2_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot2_residuals.png")

# ── PLOT 3: Feature Importances (Random Forest) ──────────────
top_n  = imp_sorted.head(15)
colors = ["#1A5276" if i == 0 else "#2980B9" if i < 4 else "#85C1E9"
          for i in range(len(top_n))]
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top_n.index, top_n.values, color=colors, edgecolor="white", height=0.65)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
ax.set_title("Random Forest — Feature Importances (Top 15)\n"
             "lag_1 removed — weather & time features now contribute meaningfully",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_feature_importances.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot3_feature_importances.png")

# ── PLOT 4: Model Comparison Bar Chart ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics_to_plot = ["MAE", "RMSE", "MAPE (%)"]
model_names     = results_df.index.tolist()
bar_colors      = [COLORS.get(n, "#AAB7B8") for n in model_names]

for ax, metric in zip(axes, metrics_to_plot):
    vals = results_df[metric].values
    bars = ax.bar(range(len(model_names)), vals, color=bar_colors,
                  edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, rotation=0)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # Highlight best bar
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

plt.suptitle("Model Comparison — Test Set 2024  (Gold border = best)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot4_model_comparison.png")


# ============================================================
# SECTION 11: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  ELECTRICITY LOAD FORECASTING — FINAL MODEL SUMMARY (Test: 2024)")
print("=" * 70)
print(f"  {'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
print("  " + "-" * 66)
for model_name, row in results_df.iterrows():
    marker = " 🏆" if model_name == best_model else ""
    print(f"  {model_name:<30} {row['MAE']:>8.2f} {row['RMSE']:>8.2f} "
          f"{row['R²']:>8.4f} {row['MAPE (%)']:>7.2f}%{marker}")
print("=" * 70)

print("""
📋 KEY IMPROVEMENTS MADE:
  ✅ lag_1 removed — eliminates 99% importance dominance
  ✅ wdir converted to sin/cos — correct circular handling
  ✅ Raw hour/month replaced with cyclical sin/cos encoding
  ✅ Removed: minute, day, year, quarter (low-value features)
  ✅ Added: is_peak_hour, is_day, temp_hour interaction
  ✅ Added: Naive baseline for reference comparison
  ✅ Added: Decision Tree, KNN, Gradient Boosting models
  ✅ Added: TimeSeriesSplit cross-validation (5 folds)
  ✅ StandardScaler applied correctly to LR and KNN only
  ✅ 4 separate publication-quality plots saved

🔬 NEXT STEPS:
  1. Try XGBoost / LightGBM for further accuracy gains
  2. Add more interaction features (temp × is_peak_hour)
  3. Explore LSTM or Transformer for sequence modelling
  4. Hyperparameter tuning with Optuna
""")