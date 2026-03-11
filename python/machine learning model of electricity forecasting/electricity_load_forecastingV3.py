# ============================================================
# MACHINE LEARNING–BASED ELECTRICITY LOAD FORECASTING
# UK National Demand | 2009–2024 | 30-min intervals
# ============================================================
# Target  : ND (National Demand, MW)
# Features: Time + Settlement Period + Renewable Generation
#           + Storage + Lag features
# Models  : Linear Regression & Random Forest Regressor
# Metrics : MAE, RMSE, R²
# ============================================================

# ── SECTION 1: Imports ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully.\n")


# ── SECTION 2: Load Dataset ─────────────────────────────────
CSV_PATH = r"C:\Users\Agrim Jain\Desktop\Coding\coding\python\machine learning model of electricity forecasting\datasets\historic_demand_2009_2024_noNaN.csv"   # ← place CSV in same folder as script

df = pd.read_csv(CSV_PATH)
print(f"📂 Raw dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Strip column names of any accidental whitespace
df.columns = df.columns.str.strip()

# Rename to clean working names
df = df.rename(columns={
    "settlement_date":          "datetime",
    "settlement_period":        "settlement_period",
    "nd":                       "load",
    "embedded_wind_generation": "wind_gen",
    "embedded_solar_generation":"solar_gen",
    "pump_storage_pumping":     "pump_storage",
    "non_bm_stor":              "non_bm_stor",
    "is_holiday":               "is_holiday",
})

# Parse datetime (format: "01-01-2009 00:00")
df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")
df = df.sort_values("datetime").reset_index(drop=True)

print(f"   Date range : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"   Total rows : {len(df):,}  (30-min intervals)")
print(f"\nLoad (ND) stats:\n{df['load'].describe().round(1)}\n")


# ── SECTION 3: Feature Engineering ──────────────────────────
print("⚙️  Engineering features …")

# -- Time features derived from datetime --
df["hour"]       = df["datetime"].dt.hour
df["minute"]     = df["datetime"].dt.minute
df["day"]        = df["datetime"].dt.day
df["month"]      = df["datetime"].dt.month
df["year"]       = df["datetime"].dt.year
df["weekday"]    = df["datetime"].dt.weekday        # 0=Mon … 6=Sun
df["weekend"]    = (df["weekday"] >= 5).astype(int)
df["quarter"]    = df["datetime"].dt.quarter

# settlement_period already in data (1–48, one per 30-min slot)

# -- Lag features (30-min intervals) --
df["lag_1"]    = df["load"].shift(1)     # 30 min ago
df["lag_2"]    = df["load"].shift(2)     # 1 hour ago
df["lag_48"]   = df["load"].shift(48)    # 24 hours ago
df["lag_336"]  = df["load"].shift(336)   # 7 days ago

# -- Rolling mean features --
df["roll_mean_4"]   = df["load"].shift(1).rolling(4).mean()    # 2-hr rolling avg
df["roll_mean_48"]  = df["load"].shift(1).rolling(48).mean()   # 24-hr rolling avg

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = [
    # Time
    "hour", "minute", "day", "month", "year",
    "weekday", "weekend", "quarter", "settlement_period",
    # Holiday flag (already in dataset)
    "is_holiday",
    # Renewable & storage generation
    "wind_gen", "solar_gen", "pump_storage", "non_bm_stor",
    # Lag features
    "lag_1", "lag_2", "lag_48", "lag_336",
    # Rolling averages
    "roll_mean_4", "roll_mean_48",
]
TARGET = "load"

# Safety check
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise KeyError(f"❌ Missing features: {missing}\n   Available: {list(df.columns)}")

print(f"✅ {len(FEATURES)} features confirmed.")
print(f"   Dataset size after lag creation: {len(df):,} rows\n")


# ── SECTION 4: Train / Test Split (chronological) ───────────
# Split at year boundary: train on 2009–2021, test on 2022–2024
SPLIT_DATE = "2022-01-01"

train_mask = df["datetime"] < SPLIT_DATE
test_mask  = df["datetime"] >= SPLIT_DATE

X_train = df.loc[train_mask, FEATURES]
y_train = df.loc[train_mask, TARGET]
X_test  = df.loc[test_mask,  FEATURES]
y_test  = df.loc[test_mask,  TARGET]
dates_test = df.loc[test_mask, "datetime"]

print(f"📊 Train: {len(X_train):,} samples  (2009-01-01 → 2021-12-31)")
print(f"   Test : {len(X_test):,}  samples  (2022-01-01 → 2024-end)\n")


# ── SECTION 5: Model Training ────────────────────────────────
print("🤖 Training models …")

# 5a. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("   ✅ Linear Regression trained.")

# 5b. Random Forest
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
    print(f"  {name:<25}  MAE={mae:7.1f} MW  RMSE={rmse:7.1f} MW  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

print("📈 Evaluation Results on Test Set (2022–2024):")
print("  " + "-" * 75)
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest",     y_test, rf_pred)
print("  " + "-" * 75)

results_df = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
print(f"\n{results_df.round(3)}\n")


# ── SECTION 7: Feature Importance ───────────────────────────
importances        = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("🌲 Random Forest Feature Importances (top 10):")
for feat, imp in importances_sorted.head(10).items():
    bar = "█" * int(imp * 60)
    print(f"  {feat:<22}  {imp:.4f}  {bar}")


# ── SECTION 8: Visualization ─────────────────────────────────
# Plot 2 weeks of test data for clarity (2 weeks × 48 slots/day)
PLOT_PERIODS = 14 * 48

idx          = np.arange(len(y_test))
plot_idx     = idx[-PLOT_PERIODS:]
plot_dates   = dates_test.values[-PLOT_PERIODS:]
plot_actual  = y_test.values[-PLOT_PERIODS:]
plot_lr      = lr_pred[-PLOT_PERIODS:]
plot_rf      = rf_pred[-PLOT_PERIODS:]

fig, axes = plt.subplots(4, 1, figsize=(18, 16))
fig.suptitle("UK National Demand Forecasting — 2009–2024\nLinear Regression vs Random Forest",
             fontsize=14, fontweight="bold", y=0.99)

# ── Plot 1: Actual vs Predicted ──
ax1 = axes[0]
ax1.plot(plot_dates, plot_actual, label="Actual ND",        color="#2C3E50", lw=1.5, zorder=3)
ax1.plot(plot_dates, plot_lr,     label="Linear Regression",color="#E74C3C", lw=1.0, linestyle="--", alpha=0.8)
ax1.plot(plot_dates, plot_rf,     label="Random Forest",    color="#27AE60", lw=1.0, alpha=0.9)
ax1.set_title("Actual vs Predicted National Demand (last 14 days of test set)", fontsize=11)
ax1.set_ylabel("National Demand (MW)")
ax1.legend(loc="upper right")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
ax1.grid(alpha=0.3)

# ── Plot 2: Residuals ──
ax2 = axes[1]
ax2.plot(plot_dates, plot_actual - plot_lr, label="LR Residuals", color="#E74C3C", lw=0.7, alpha=0.8)
ax2.plot(plot_dates, plot_actual - plot_rf, label="RF Residuals", color="#27AE60", lw=0.7, alpha=0.8)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Residuals (Actual − Predicted)", fontsize=11)
ax2.set_ylabel("Residual (MW)")
ax2.legend(loc="upper right")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
ax2.grid(alpha=0.3)

# # ── Plot 3: Feature Importances ──
# ax3 = axes[2]
# top_n   = importances_sorted.head(12)
# colors  = ["#2980B9" if i < 3 else "#85C1E9" for i in range(len(top_n))]
# bars    = ax3.barh(top_n.index, top_n.values, color=colors, edgecolor="white")
# ax3.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
# ax3.set_title("Random Forest — Top 12 Feature Importances", fontsize=11)
# ax3.set_xlabel("Importance Score")
# ax3.invert_yaxis()
# ax3.grid(axis="x", alpha=0.3)

# ── Plot 4: Model comparison bar chart ──
ax4 = axes[3]
metrics    = ["MAE", "RMSE", "MAPE"]
lr_vals    = [lr_metrics["MAE"], lr_metrics["RMSE"], lr_metrics["MAPE"] * 10]
rf_vals    = [rf_metrics["MAE"], rf_metrics["RMSE"], rf_metrics["MAPE"] * 10]
x          = np.arange(len(metrics))
w          = 0.35
bars1 = ax4.bar(x - w/2, lr_vals, w, label="Linear Regression", color="#E74C3C", alpha=0.85)
bars2 = ax4.bar(x + w/2, rf_vals, w, label="Random Forest",     color="#27AE60", alpha=0.85)
ax4.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
ax4.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
ax4.set_xticks(x)
ax4.set_xticklabels(["MAE (MW)", "RMSE (MW)", "MAPE×10 (%)"])
ax4.set_title("Model Error Comparison", fontsize=11)
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("load_forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊 Plot saved to 'load_forecast_results.png'")


# ── SECTION 9: Summary ───────────────────────────────────────
print("\n" + "=" * 65)
print("  UK NATIONAL DEMAND FORECASTING — BASELINE MODEL SUMMARY")
print("=" * 65)
for _, row in results_df.iterrows():
    print(f"  {row.name:<25}  MAE={row.MAE:.1f} MW  RMSE={row.RMSE:.1f} MW  "
          f"R²={row.R2:.4f}  MAPE={row.MAPE:.2f}%")
print("=" * 65)
print("""
Next steps to improve accuracy:
  1. Add XGBoost / LightGBM (major accuracy boost expected)
  2. Add more lag steps: lag_96 (2 days), lag_672 (2 weeks)
  3. Add weather data (temperature is the #1 demand driver)
  4. Tune hyperparameters with Optuna or GridSearchCV
  5. Explore LSTM / Transformer for sequence modelling
""")