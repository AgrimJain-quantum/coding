"""Top-level orchestration for the modular v12 forecasting pipeline."""

from .config import CSV_PATH
from .data import engineer_features, load_and_clean_data, split_and_scale
from .ensemble import build_weighted_ensemble
from .environment import configure_runtime
from .lstm import train_lstm
from .models import (
    get_default_models,
    get_feature_importances,
    print_feature_importances,
    run_xgboost_time_series_cv,
    train_default_models,
)
from .summary import print_final_summary
from .tuning import tune_lightgbm, tune_xgboost
from .visualization import create_all_plots


def main(csv_path=CSV_PATH) -> None:
    """Run the complete electricity load forecasting pipeline."""
    configure_runtime()

    df = load_and_clean_data(csv_path)
    df = engineer_features(df)
    prepared = split_and_scale(df)

    models = get_default_models()
    all_metrics, all_preds, results_df, best_model = train_default_models(prepared, models)

    run_xgboost_time_series_cv(df)

    xgb_tuning = tune_xgboost(
        prepared.X_train,
        prepared.y_train,
        prepared.X_test,
        prepared.y_test,
        all_metrics,
        all_preds,
    )
    results_df = xgb_tuning["results_df"]
    best_model = xgb_tuning["best_model"]

    lgbm_tuning = tune_lightgbm(
        prepared.X_train,
        prepared.y_train,
        prepared.X_test,
        prepared.y_test,
        all_metrics,
        all_preds,
    )
    results_df = lgbm_tuning["results_df"]
    best_model = lgbm_tuning["best_model"]

    ensemble = build_weighted_ensemble(
        prepared.y_test,
        xgb_tuning["preds"],
        lgbm_tuning["preds"],
        all_metrics,
        all_preds,
        results_df,
    )
    results_df = ensemble["results_df"]
    best_model = ensemble["best_model"]

    lstm = train_lstm(df, prepared.y_test, all_metrics, all_preds)
    results_df = lstm["results_df"]
    best_model = lstm["best_model"]

    rf_imp, xgb_imp, lgbm_imp = get_feature_importances(models)
    print_feature_importances(rf_imp, xgb_imp, lgbm_imp)

    create_all_plots(
        prepared.dates_test,
        prepared.y_test,
        all_preds,
        results_df,
        rf_imp,
        xgb_imp,
        lgbm_imp,
        xgb_tuning["trial_nums"],
        xgb_tuning["trial_maes"],
        xgb_tuning["running_min"],
        lgbm_tuning["trial_nums"],
        lgbm_tuning["trial_maes"],
        lgbm_tuning["running_min"],
        ensemble["w_xgb"],
        ensemble["w_lgbm"],
        lstm["history"],
    )

    print_final_summary(results_df, best_model)

