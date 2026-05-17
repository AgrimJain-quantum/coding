"""Final textual summary for the v12 pipeline."""


def print_final_summary(results_df, best_model: str) -> None:
    """Print the final sorted model table and changelog."""
    print("\n" + "=" * 72)
    print("  ELECTRICITY LOAD FORECASTING v12 - FINAL SUMMARY (Test: 2024)")
    print("=" * 72)
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE%':>8}")
    print("  " + "-" * 62)
    for model_name, row in results_df.iterrows():
        marker = "  BEST" if model_name == best_model else ""
        print(
            f"  {model_name:<25} {row['MAE']:>8.2f} {row['RMSE']:>8.2f} "
            f"{row['R2']:>8.4f} {row['MAPE (%)']:>7.2f}%{marker}"
        )
    print("=" * 72)

    print(
        """
v12 CHANGELOG:
  - LSTM added (Section 8e)
      * 2-layer stacked LSTM: 128 -> 64 units + Dropout(0.2)
      * Lookback = 24 steps (2-hour context window)
      * MinMaxScaler on all features + target (no leakage)
      * EarlyStopping (patience=5) + ReduceLROnPlateau (patience=3)
      * Full chronological train/test integrity preserved
  - Plot 12: LSTM vs Ensemble vs XGBoost (Tuned) - last 2 days
  - Plot 13: LSTM training loss curves (train vs val + best epoch)
  - CSV_PATH kept for Kaggle, with local dataset fallback when present
  - LSTM color added to global COLORS dict (#D4145A)
"""
    )

