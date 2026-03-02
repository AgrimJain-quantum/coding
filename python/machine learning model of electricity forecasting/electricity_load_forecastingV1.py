# ============================================================
# MACHINE LEARNING–BASED ELECTRICITY LOAD FORECASTING
# Baseline Regression Model | Residential/Urban Sector
# ============================================================
# Features: Historical load + Time-based features only
# Models   : Linear Regression & Random Forest Regressor
# Metrics  : MAE, RMSE, R²
# ============================================================

# ── SECTION 1: Imports ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully.\n")


# ── SECTION 2: Load Dataset ─────────────────────────────────
# Expected CSV format:
#   datetime  : e.g. "2023-01-01 00:00:00"  (parseable by pandas)
#   load      : numerical electricity load in kWh or MW
#
# To use your own file, replace the path below.
# A synthetic dataset is generated automatically if the file
# is not found, so you can run the script immediately.

CSV_PATH = "load_forecasting_dataset_corrected.csv"   # ← change to your file path

try:
    df = pd.read_csv(CSV_PATH, parse_dates=["Timestamp"])
    df = df.rename(columns={"Timestamp": "datetime", "Load Demand (kW)": "load"})
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"📂 Loaded dataset: {len(df):,} rows from '{CSV_PATH}'")

except FileNotFoundError:
    print(f"⚠️  '{CSV_PATH}' not found – generating a synthetic dataset for demonstration.\n")

    # ── Synthetic data: 2 years of hourly load with realistic patterns ──
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=17_520, freq="h")   # 2 years

    hour     = dates.hour
    month    = dates.month
    weekday  = dates.weekday

    # Base load with daily, weekly, and seasonal cycles
    daily_cycle    = 10 * np.sin(2 * np.pi * (hour - 6) / 24)
    weekly_factor  = np.where(weekday < 5, 1.0, 0.85)              # lower on weekends
    seasonal_cycle = 8  * np.sin(2 * np.pi * (month - 1) / 12)    # summer/winter peaks
    noise          = np.random.normal(0, 3, len(dates))

    load = 100 + daily_cycle + seasonal_cycle + noise
    load *= weekly_factor
    load = np.clip(load, 40, 180)                                   # realistic bounds

    df = pd.DataFrame({"datetime": dates, "load": load})
    print(f"✅ Synthetic dataset created: {len(df):,} rows (2 years, hourly)\n")

print(df.head())
print(f"\nLoad stats:\n{df['load'].describe().round(2)}")


# ── SECTION 3: Feature Engineering ──────────────────────────
print("\n⚙️  Engineering time-based features …")

df["hour"]    = df["datetime"].dt.hour
df["day"]     = df["datetime"].dt.day
df["month"]   = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday          # 0 = Monday … 6 = Sunday
df["weekend"] = (df["weekday"] >= 5).astype(int)   # 1 if Saturday or Sunday

# Optional: lag features (previous hour load) – useful for capturing autocorrelation
df["lag_1h"]  = df["load"].shift(1)   # 15-min intervals → shift(1) = 15 mins ago
df["lag_4h"]  = df["load"].shift(4)   # 1 hour ago (4 × 15 min)
df["lag_96h"] = df["load"].shift(96)  # 24 hours ago (96 × 15 min)

df.dropna(inplace=True)   # remove rows with NaN from lag creation
df.reset_index(drop=True, inplace=True)



FEATURES = [
    # Time-based (pre-built in your dataset)
    "Hour of Day", "Day of Week", "Month", "Season", "Public Event",

    # Weather
    "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)",
    "Rainfall (mm)", "Solar Irradiance (W/m²)",

    # Economic
    "GDP (USD)", "Per Capita Energy Use (kWh)", "Electricity Price (LKR/kWh)",

    # Lag features (autocorrelation)
    "lag_1h", "lag_4h", "lag_96h"
]
TARGET   = "load"

print(f"✅ Features used: {FEATURES}")
print(f"   Dataset size after lag creation: {len(df):,} rows\n")


# ── SECTION 4: Train / Test Split ───────────────────────────
# Time-series data must be split chronologically (no shuffling).

SPLIT_RATIO = 0.80   # 80 % train | 20 % test

split_idx = int(len(df) * SPLIT_RATIO)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

dates_test = df["datetime"].iloc[split_idx:]

print(f"📊 Train size : {len(X_train):,} samples")
print(f"   Test  size : {len(X_test):,} samples")
print(f"   Train period: {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[split_idx-1].date()}")
print(f"   Test  period: {df['datetime'].iloc[split_idx].date()} → {df['datetime'].iloc[-1].date()}\n")


# ── SECTION 5: Model Training ────────────────────────────────
print("🤖 Training models …")

# ── 5a. Linear Regression ──
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("   ✅ Linear Regression trained.")

# ── 5b. Random Forest ──   
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("   ✅ Random Forest trained.\n")


# ── SECTION 6: Evaluation ────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {'Model':<22} MAE={mae:6.3f}  RMSE={rmse:6.3f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

print("📈 Evaluation Results on Test Set:")
print("  " + "-" * 58)
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest",     y_test, rf_pred)
print("  " + "-" * 58)

results_df = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
print(f"\n{results_df.round(4)}\n")


# ── SECTION 7: Feature Importance (Random Forest) ───────────
importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("🌲 Random Forest Feature Importances:")
for feat, imp in importances_sorted.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<12}  {imp:.4f}  {bar}")


# ── SECTION 8: Visualization ─────────────────────────────────
PLOT_DAYS = 14   # show last N days of test period for clarity
hours_to_plot = PLOT_DAYS * 24

plot_dates  = dates_test.values[-hours_to_plot:]
plot_actual = y_test.values[-hours_to_plot:]
plot_lr     = lr_pred[-hours_to_plot:]
plot_rf     = rf_pred[-hours_to_plot:]

fig, axes = plt.subplots(3, 1, figsize=(16, 13))
fig.suptitle("Electricity Load Forecasting — Baseline Models", fontsize=15, fontweight="bold", y=0.98)

# ── Plot 1: Actual vs Predicted (both models) ──
ax1 = axes[0]
ax1.plot(plot_dates, plot_actual, label="Actual Load",       color="#2C3E50", lw=1.5, zorder=3)
ax1.plot(plot_dates, plot_lr,     label="Linear Regression", color="#E74C3C", lw=1.2, linestyle="--", alpha=0.85)
ax1.plot(plot_dates, plot_rf,     label="Random Forest",     color="#27AE60", lw=1.2, linestyle="-",  alpha=0.90)
ax1.set_title(f"Actual vs Predicted Load (last {PLOT_DAYS} days of test set)", fontsize=12)
ax1.set_ylabel("Load (kWh / MW)")
ax1.legend(loc="upper right")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.grid(alpha=0.3)

# ── Plot 2: Residuals ──
ax2 = axes[1]
ax2.plot(plot_dates, plot_actual - plot_lr, label="LR Residuals",  color="#E74C3C", lw=0.9, alpha=0.8)
ax2.plot(plot_dates, plot_actual - plot_rf, label="RF Residuals",  color="#27AE60", lw=0.9, alpha=0.8)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Residuals (Actual − Predicted)", fontsize=12)
ax2.set_ylabel("Residual")
ax2.legend(loc="upper right")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.grid(alpha=0.3)

# ── Plot 3: Feature Importances ──
ax3 = axes[2]
colors = ["#3498DB" if i == 0 else "#85C1E9" for i in range(len(importances_sorted))]
bars = ax3.barh(importances_sorted.index, importances_sorted.values, color=colors, edgecolor="white")
ax3.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax3.set_title("Random Forest — Feature Importances", fontsize=12)
ax3.set_xlabel("Importance Score")
ax3.invert_yaxis()
ax3.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("load_forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊 Plot saved to 'load_forecast_results.png'")


# ── SECTION 9: Summary ───────────────────────────────────────
print("\n" + "=" * 58)
print("  BASELINE MODEL SUMMARY")
print("=" * 58)
for _, row in results_df.iterrows():
    print(f"  {row.name:<25}  MAE={row.MAE:.3f}  RMSE={row.RMSE:.3f}  R²={row.R2:.4f}")
print("=" * 58)
print("""
Next steps to improve accuracy:
  1. Add weather features (temperature, humidity, wind)
  2. Add holiday / festival flags
  3. Try Gradient Boosting (XGBoost / LightGBM)
  4. Explore LSTM or Transformer-based time-series models
  5. Tune hyperparameters with GridSearchCV / Optuna
""")
