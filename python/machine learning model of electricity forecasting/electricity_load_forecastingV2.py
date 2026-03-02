# ============================================================
# MACHINE LEARNING–BASED ELECTRICITY LOAD FORECASTING
# Baseline Regression Model | Residential/Urban Sector
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
CSV_PATH = r"C:\Users\Agrim Jain\Desktop\Coding\coding\python\machine learning model of electricity forecasting\load_forecasting_dataset_corrected.csv"   # ← your file path

df = pd.read_csv(CSV_PATH)

# Step 2a: Inspect actual column names (reveals hidden spaces/encoding issues)
print("📋 Raw column names detected:")
for col in df.columns:
    print(f"   {repr(col)}")

# Step 2b: Strip whitespace from all column names
df.columns = df.columns.str.strip()

# Step 2c: Rename to clean internal names
df = df.rename(columns={
    "Timestamp":                   "datetime",
    "Load Demand (kW)":            "load",
    "Temperature (°C)":            "temperature",
    "Humidity (%)":                "humidity",
    "Wind Speed (m/s)":            "wind_speed",
    "Rainfall (mm)":               "rainfall",
    "Solar Irradiance (W/m²)":     "solar_irradiance",
    "GDP (LKR)":                   "gdp",
    "Per Capita Energy Use (kWh)": "per_capita_energy",
    "Electricity Price (LKR/kWh)": "electricity_price",
    "Day of Week":                 "day_of_week",
    "Hour of Day":                 "hour_of_day",
    "Month":                       "month",
    "Season":                      "season",
    "Public Event":                "public_event",
})

# Step 2d: Parse datetime and sort
df["datetime"] = pd.to_datetime(df["datetime"], format="%m-%d-%Y %H:%M")
df = df.sort_values("datetime").reset_index(drop=True)

print(f"\n📂 Loaded: {len(df):,} rows")
print(f"   Columns: {list(df.columns)}")
print(f"\n{df.head(3)}")
print(f"\nLoad stats:\n{df['load'].describe().round(2)}")


# ── SECTION 3: Feature Engineering ──────────────────────────
print("\n⚙️  Engineering features …")

# Encode Season text → integer
season_map = {"Summer": 0, "Spring": 1, "Fall": 2, "Autumn": 2, "Winter": 3}
df["season"] = df["season"].map(season_map)

if df["season"].isna().any():
    print("⚠️  Some Season values not in map — filling with mode.")
    df["season"] = df["season"].fillna(df["season"].mode()[0])

# Lag features (15-min intervals)
df["lag_1step"]  = df["load"].shift(1)    # 15 minutes ago
df["lag_4step"]  = df["load"].shift(4)    # 1 hour ago
df["lag_96step"] = df["load"].shift(96)   # 24 hours ago

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = [
    "hour_of_day", "day_of_week", "month", "season", "public_event",
    "temperature", "humidity", "wind_speed", "rainfall", "solar_irradiance",
    "gdp", "per_capita_energy", "electricity_price",
    "lag_1step", "lag_4step", "lag_96step",
]
TARGET = "load"

# Safety check
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"\n❌ Still missing columns: {missing}")
    print(f"   Available columns: {list(df.columns)}")
    raise KeyError(f"Fix column mapping for: {missing}")

print(f"✅ All {len(FEATURES)} features confirmed present.")
print(f"   Dataset size after lag creation: {len(df):,} rows\n")


# ── SECTION 4: Train / Test Split (chronological) ───────────
SPLIT_RATIO = 0.80
split_idx   = int(len(df) * SPLIT_RATIO)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test      = df["datetime"].iloc[split_idx:]

print(f"📊 Train: {len(X_train):,} samples  ({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[split_idx-1].date()})")
print(f"   Test : {len(X_test):,}  samples  ({df['datetime'].iloc[split_idx].date()} → {df['datetime'].iloc[-1].date()})\n")


# ── SECTION 5: Model Training ────────────────────────────────
print("🤖 Training models …")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("   ✅ Linear Regression trained.")

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
    print(f"  {name:<25}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

print("📈 Evaluation Results on Test Set:")
print("  " + "-" * 65)
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest",     y_test, rf_pred)
print("  " + "-" * 65)

results_df = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
print(f"\n{results_df.round(4)}\n")


# ── SECTION 7: Feature Importance ───────────────────────────
importances        = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("🌲 Random Forest Feature Importances:")
for feat, imp in importances_sorted.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<22}  {imp:.4f}  {bar}")


# ── SECTION 8: Visualization ─────────────────────────────────
PLOT_PERIODS = 14 * 96   # last 14 days of 15-min data

plot_dates  = dates_test.values[-PLOT_PERIODS:]
plot_actual = y_test.values[-PLOT_PERIODS:]
plot_lr     = lr_pred[-PLOT_PERIODS:]
plot_rf     = rf_pred[-PLOT_PERIODS:]

fig, axes = plt.subplots(3, 1, figsize=(16, 13))
fig.suptitle("Electricity Load Forecasting — Baseline Models", fontsize=15, fontweight="bold", y=0.98)

ax1 = axes[0]
ax1.plot(plot_dates, plot_actual, label="Actual Load",       color="#2C3E50", lw=1.5, zorder=3)
ax1.plot(plot_dates, plot_lr,     label="Linear Regression", color="#E74C3C", lw=1.0, linestyle="--", alpha=0.85)
ax1.plot(plot_dates, plot_rf,     label="Random Forest",     color="#27AE60", lw=1.0, alpha=0.90)
ax1.set_title("Actual vs Predicted Load (last 14 days of test set)", fontsize=12)
ax1.set_ylabel("Load (kW)")
ax1.legend(loc="upper right")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(plot_dates, plot_actual - plot_lr, label="LR Residuals", color="#E74C3C", lw=0.8, alpha=0.8)
ax2.plot(plot_dates, plot_actual - plot_rf, label="RF Residuals", color="#27AE60", lw=0.8, alpha=0.8)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Residuals (Actual − Predicted)", fontsize=12)
ax2.set_ylabel("Residual (kW)")
ax2.legend(loc="upper right")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.grid(alpha=0.3)

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
print("\n" + "=" * 60)
print("  BASELINE MODEL SUMMARY")
print("=" * 60)
for _, row in results_df.iterrows():
    print(f"  {row.name:<25}  MAE={row.MAE:.3f}  RMSE={row.RMSE:.3f}  R²={row.R2:.4f}")
print("=" * 60)
print("""
Next steps to improve accuracy:
  1. Try Gradient Boosting (XGBoost / LightGBM)
  2. Tune hyperparameters with GridSearchCV / Optuna
  3. Explore LSTM or Transformer-based time-series models
  4. Add interaction features (e.g. temp × hour_of_day)
  5. Add rolling mean / rolling std lag features
""")