"""Default model definitions, training, cross-validation, and importances."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import FEATURES, RANDOM_STATE, TARGET
from .data import PreparedData
from .metrics import best_model_name, build_results_frame, compute_metrics


def get_default_models() -> dict:
    """Create the default model set from the original v12 script."""
    return {
        "Linear Regression": (LinearRegression(), "scaled"),
        "Decision Tree": (
            DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE),
            "raw",
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "raw",
        ),
        "KNN": (
            KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
            "scaled",
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ),
            "raw",
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            "raw",
        ),
        "LightGBM": (
            lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "raw",
        ),
    }


def _print_metric_line(name: str, metrics: dict) -> None:
    print(
        f"  OK {name:<22} MAE={metrics['MAE']:7.2f}  "
        f"RMSE={metrics['RMSE']:7.2f}  R2={metrics['R2']:.4f}  "
        f"MAPE={metrics['MAPE (%)']:.2f}%"
    )


def train_default_models(prepared: PreparedData, models: dict) -> tuple[list[dict], dict, pd.DataFrame, str]:
    """Train the naive baseline plus all default machine-learning models."""
    all_metrics = []
    all_preds = {}

    naive_pred = prepared.X_test["lag_288"].values
    print("Naive baseline: prediction = load(t-288) [same time yesterday]\n")
    print("Training models ...\n")

    metrics = compute_metrics("Naive Baseline", prepared.y_test, naive_pred)
    all_metrics.append(metrics)
    all_preds["Naive Baseline"] = naive_pred
    _print_metric_line("Naive Baseline", metrics)

    for name, (model, data_type) in models.items():
        X_tr = prepared.X_train_sc if data_type == "scaled" else prepared.X_train
        X_te = prepared.X_test_sc if data_type == "scaled" else prepared.X_test

        model.fit(X_tr, prepared.y_train)
        preds = model.predict(X_te)

        metrics = compute_metrics(name, prepared.y_test, preds)
        all_metrics.append(metrics)
        all_preds[name] = preds
        _print_metric_line(name, metrics)

    results_df = build_results_frame(all_metrics)
    best_model = best_model_name(results_df)

    print("\n" + "=" * 72)
    print("  BASELINE RESULTS - sorted by RMSE ascending")
    print("=" * 72)
    print(results_df.to_string())
    print("=" * 72)
    print(f"\n  Best so far: {best_model}\n")

    return all_metrics, all_preds, results_df, best_model


def run_xgboost_time_series_cv(df: pd.DataFrame) -> list[float]:
    """Run the original 5-fold TimeSeriesSplit CV for XGBoost."""
    print("TimeSeriesSplit cross-validation on XGBoost (5 folds) ...")

    xgb_cv = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    X_full = df[FEATURES]
    y_full = df[TARGET]

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_full), 1):
        xgb_cv.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
        preds = xgb_cv.predict(X_full.iloc[te_idx])
        mae = mean_absolute_error(y_full.iloc[te_idx], preds)
        cv_maes.append(mae)
        print(f"   Fold {fold}: MAE = {mae:.2f} MW")

    print(f"\n   CV Mean MAE : {np.mean(cv_maes):.2f} MW")
    print(f"   CV Std  MAE : {np.std(cv_maes):.2f} MW\n")
    return cv_maes


def get_feature_importances(models: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return feature-importance series for the fitted tree models."""
    rf_model = models["Random Forest"][0]
    xgb_model = models["XGBoost"][0]
    lgbm_model = models["LightGBM"][0]

    rf_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    lgbm_imp = pd.Series(lgbm_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return rf_imp, xgb_imp, lgbm_imp


def print_feature_importances(rf_imp: pd.Series, xgb_imp: pd.Series, lgbm_imp: pd.Series) -> None:
    """Print the top 10 importances in the same place as the original script."""
    for title, importances in [
        ("Random Forest", rf_imp),
        ("XGBoost", xgb_imp),
        ("LightGBM", lgbm_imp),
    ]:
        print(f"\n{title} - Top 10:")
        for feature, importance in importances.head(10).items():
            print(f"  {feature:<22}  {importance:.4f}  {'#' * int(importance * 80)}")

