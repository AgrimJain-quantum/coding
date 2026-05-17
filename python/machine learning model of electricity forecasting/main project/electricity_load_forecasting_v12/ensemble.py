"""Weighted ensemble helper."""

from .metrics import best_model_name, build_results_frame, compute_metrics


def build_weighted_ensemble(y_test, xgb_preds, lgbm_preds, all_metrics, all_preds, results_df) -> dict:
    """Blend tuned XGBoost and tuned LightGBM with inverse-RMSE weights."""
    print("Building Ensemble: XGBoost (Tuned) + LightGBM (Tuned) ...")

    xgb_rmse = results_df.loc["XGBoost (Tuned)", "RMSE"]
    lgbm_rmse = results_df.loc["LightGBM (Tuned)", "RMSE"]

    inv_xgb = 1.0 / xgb_rmse
    inv_lgbm = 1.0 / lgbm_rmse
    total = inv_xgb + inv_lgbm
    w_xgb = inv_xgb / total
    w_lgbm = inv_lgbm / total

    print(f"   XGBoost (Tuned)  weight : {w_xgb:.4f}  (RMSE={xgb_rmse:.2f} MW)")
    print(f"   LightGBM (Tuned) weight : {w_lgbm:.4f}  (RMSE={lgbm_rmse:.2f} MW)")

    ensemble_preds = w_xgb * xgb_preds + w_lgbm * lgbm_preds
    metrics = compute_metrics("Ensemble (XGB+LGBM)", y_test, ensemble_preds)
    all_metrics.append(metrics)
    all_preds["Ensemble (XGB+LGBM)"] = ensemble_preds

    results_df = build_results_frame(all_metrics)
    best_model = best_model_name(results_df)

    print("\nComponent vs Ensemble - Test 2024")
    print(f"   {'Model':<25}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'MAPE%':>8}")
    print(f"   {'-' * 62}")
    for model_name in ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"]:
        row = results_df.loc[model_name]
        marker = "  <- best" if model_name == best_model else ""
        print(
            f"   {model_name:<25}  {row['MAE']:>8.2f}  {row['RMSE']:>8.2f}  "
            f"{row['R2']:>8.4f}  {row['MAPE (%)']:>7.2f}%{marker}"
        )

    return {
        "preds": ensemble_preds,
        "w_xgb": w_xgb,
        "w_lgbm": w_lgbm,
        "results_df": results_df,
        "best_model": best_model,
    }

