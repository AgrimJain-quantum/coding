# ============================================================
# MACHINE LEARNING–BASED ELECTRICITY LOAD FORECASTING
# Power Demand | 2021–2024 | 5-minute intervals
# ============================================================
# Target  : Power demand (MW)
# Features: Time + Weather + Lag + Rolling features
# Models  : Linear Regression & Random Forest Regressor
# Metrics : MAE, RMSE, R², MAPE
# ============================================================

# ── SECTION 1: Imports ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully.\n")


# ── SECTION 2: Load Dataset ─────────────────────────────────
CSV_PATH = r"C:\Users\Agrim Jain\Desktop\Coding\coding\python\machine learning model of electricity forecasting\datasets\powerdemand_5min_2021_to_2024_with weather.csv"

df = pd.read_csv(CSV_PATH)
print(f"📂 Raw dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop unnamed index column
df.drop(columns=["Unnamed: 0"], inplace=True)

# Strip column whitespace
df.columns = df.columns.str.strip()

# Rename target to clean name
df = df.rename(columns={"Power demand": "load"})

# Parse datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print(f"   Date range : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"   Total rows : {len(df):,}  (5-min intervals)")
print(f"\nLoad (Power Demand) stats:\n{df['load'].describe().round(2)}\n")


# ── SECTION 3: Feature Engineering ──────────────────────────
print("⚙️  Engineering features …")

# Fix missing wind direction with forward fill (only 540 rows affected)
df["wdir"] = df["wdir"].ffill()

# Drop the pre-computed moving_avg_3 — we'll build our own lag/rolling features
df.drop(columns=["moving_avg_3"], inplace=True)

# ── Time features (year/month/day/hour/minute already in data) ──
df["weekday"]  = df["datetime"].dt.weekday       # 0=Mon … 6=Sun
df["weekend"]  = (df["weekday"] >= 5).astype(int)
df["quarter"]  = df["datetime"].dt.quarter

# Cyclical encoding for hour and month (helps models understand circular time)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]= np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]= np.cos(2 * np.pi * df["month"] / 12)

# ── Lag features (5-min intervals) ──
df["lag_1"]    = df["load"].shift(1)      # 5 min ago
df["lag_12"]   = df["load"].shift(12)     # 1 hour ago
df["lag_288"]  = df["load"].shift(288)    # 24 hours ago
df["lag_2016"] = df["load"].shift(2016)   # 7 days ago

# ── Rolling statistics ──
df["roll_mean_12"]  = df["load"].shift(1).rolling(12).mean()   # 1-hr rolling mean
df["roll_mean_288"] = df["load"].shift(1).rolling(288).mean()  # 24-hr rolling mean
df["roll_std_12"]   = df["load"].shift(1).rolling(12).std()    # 1-hr rolling std
df["roll_max_12"]   = df["load"].shift(1).rolling(12).max()    # 1-hr rolling max
df["roll_min_12"]   = df["load"].shift(1).rolling(12).min()    # 1-hr rolling min

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = [
    # Time
    "hour", "minute", "day", "month", "year",
    "weekday", "weekend", "quarter",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Weather
    "temp", "dwpt", "rhum", "wdir", "wspd", "pres",
    # Lag features
    "lag_1", "lag_12", "lag_288", "lag_2016",
    # Rolling features
    "roll_mean_12", "roll_mean_288",
    "roll_std_12", "roll_max_12", "roll_min_12",
]
TARGET = "load"

# Safety check
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise KeyError(f"❌ Missing features: {missing}")

print(f"✅ {len(FEATURES)} features confirmed.")
print(f"   Dataset size after feature engineering: {len(df):,} rows\n")


# ── SECTION 4: Train / Test Split (chronological) ───────────
# Train: 2021–2023  |  Test: 2024
SPLIT_DATE = "2024-01-01"

train_mask = df["datetime"] < SPLIT_DATE
test_mask  = df["datetime"] >= SPLIT_DATE

X_train    = df.loc[train_mask, FEATURES]
y_train    = df.loc[train_mask, TARGET]
X_test     = df.loc[test_mask,  FEATURES]
y_test     = df.loc[test_mask,  TARGET]
dates_test = df.loc[test_mask,  "datetime"]

print(f"📊 Train: {len(X_train):,} samples  (2021 → 2023)")
print(f"   Test : {len(X_test):,}  samples  (2024)\n")


# ── SECTION 5: Model Training ────────────────────────────────
print("🤖 Training models …")

# Scale features for Linear Regression
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 5a. Linear Regression (uses scaled data)
lr_model = LinearRegression()
lr_model.fit(X_train_sc, y_train)
lr_pred = lr_model.predict(X_test_sc)
print("   ✅ Linear Regression trained.")

# 5b. Random Forest (uses raw data — tree models don't need scaling)
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42,
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("   ✅ Random Forest trained.\n")


# ── SECTION 6: Evaluation ────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {name:<25}  MAE={mae:7.2f} MW  RMSE={rmse:7.2f} MW  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

print("📈 Evaluation Results on Test Set (2024):")
print("  " + "-" * 78)
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest",     y_test, rf_pred)
print("  " + "-" * 78)

results_df = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
print(f"\n{results_df.round(3)}\n")


# ── SECTION 7: Feature Importance ───────────────────────────
importances        = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("🌲 Random Forest Feature Importances (top 12):")
for feat, imp in importances_sorted.head(12).items():
    bar = "█" * int(imp * 60)
    print(f"  {feat:<20}  {imp:.4f}  {bar}")


# ── SECTION 8: Visualization (4 separate plots) ──────────────
# Show last 7 days of test data (7 × 24hr × 12 per hr = 2016 periods)
PLOT_PERIODS = 7 * 288

plot_dates  = dates_test.values[-PLOT_PERIODS:]
plot_actual = y_test.values[-PLOT_PERIODS:]
plot_lr     = lr_pred[-PLOT_PERIODS:]
plot_rf     = rf_pred[-PLOT_PERIODS:]

# ── Plot 1: Actual vs Predicted ──────────────────────────────
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(plot_dates, plot_actual, label="Actual Demand",     color="#2C3E50", lw=1.5, zorder=3)
ax.plot(plot_dates, plot_lr,     label="Linear Regression", color="#E74C3C", lw=1.0, linestyle="--", alpha=0.8)
ax.plot(plot_dates, plot_rf,     label="Random Forest",     color="#27AE60", lw=1.0, alpha=0.9)
ax.set_title("Actual vs Predicted Power Demand — Last 7 Days of Test Set (2024)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot1_actual_vs_predicted.png")

# ── Plot 2: Residuals ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(plot_dates, plot_actual - plot_lr, label="LR Residuals", color="#E74C3C", lw=0.7, alpha=0.8)
ax.plot(plot_dates, plot_actual - plot_rf, label="RF Residuals", color="#27AE60", lw=0.7, alpha=0.8)
ax.axhline(0, color="black", lw=0.9, linestyle="--")
ax.fill_between(plot_dates, plot_actual - plot_rf, 0,
                where=(plot_actual - plot_rf) > 0, alpha=0.08, color="#27AE60")
ax.fill_between(plot_dates, plot_actual - plot_rf, 0,
                where=(plot_actual - plot_rf) < 0, alpha=0.08, color="#E74C3C")
ax.set_title("Residuals (Actual − Predicted)", fontsize=13, fontweight="bold")
ax.set_ylabel("Residual (MW)")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot2_residuals.png")

# ── Plot 3: Feature Importances ───────────────────────────────
top_n  = importances_sorted.head(15)
colors = ["#1A5276" if i == 0 else "#2980B9" if i < 3 else "#85C1E9"
          for i in range(len(top_n))]
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top_n.index, top_n.values, color=colors, edgecolor="white", height=0.7)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
ax.set_title("Random Forest — Top 15 Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_feature_importances.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot3_feature_importances.png")

# ── Plot 4: Model Error Comparison ────────────────────────────
metrics = ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]
lr_vals = [lr_metrics["MAE"], lr_metrics["RMSE"], lr_metrics["MAPE"]]
rf_vals = [rf_metrics["MAE"], rf_metrics["RMSE"], rf_metrics["MAPE"]]
x, w    = np.arange(len(metrics)), 0.32

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - w/2, lr_vals, w, label="Linear Regression", color="#E74C3C", alpha=0.87)
bars2 = ax.bar(x + w/2, rf_vals, w, label="Random Forest",     color="#27AE60", alpha=0.87)
ax.bar_label(bars1, fmt="%.2f", padding=4, fontsize=10, fontweight="bold")
ax.bar_label(bars2, fmt="%.2f", padding=4, fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_title("Model Error Comparison — Test Set 2024", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot4_model_comparison.png")


# ── SECTION 9: Summary ───────────────────────────────────────
print("\n" + "=" * 68)
print("  POWER DEMAND FORECASTING — BASELINE MODEL SUMMARY (Test: 2024)")
print("=" * 68)
for _, row in results_df.iterrows():
    print(f"  {row.name:<25}  MAE={row.MAE:.2f} MW  RMSE={row.RMSE:.2f} MW  "
          f"R²={row.R2:.4f}  MAPE={row.MAPE:.2f}%")
print("=" * 68)
print("""
Next steps to improve accuracy:
  1. Add XGBoost / LightGBM — biggest single accuracy jump
  2. Add more lag steps: lag_4032 (14 days)
  3. Add interaction feature: temp × hour (captures cooling load peaks)
  4. Tune hyperparameters with Optuna or GridSearchCV
  5. Explore LSTM / Transformer for deep sequence modelling
""")