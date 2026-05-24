"""Metric helpers shared by the training stages."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(name, y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R2, and MAPE for one model."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape}


def build_results_frame(all_metrics: list[dict]) -> pd.DataFrame:
    """Return the sorted metrics table used throughout the pipeline."""
    return pd.DataFrame(all_metrics).set_index("Model").round(3).sort_values("RMSE")


def best_model_name(results_df: pd.DataFrame) -> str:
    """Return the current best model by RMSE."""
    return str(results_df.index[0])

