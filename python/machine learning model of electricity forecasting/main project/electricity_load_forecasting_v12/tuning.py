"""Optuna tuning for XGBoost and LightGBM."""

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .config import LGBM_CV, LGBM_TRIALS, OPTUNA_CV, OPTUNA_TRIALS, RANDOM_STATE
from .metrics import best_model_name, build_results_frame, compute_metrics


def _plot_optuna_convergence(trial_nums, trial_maes, running_min, color, title, filename) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(trial_nums, trial_maes, color="#95A5A6", s=18, alpha=0.6, label="Trial MAE")
    ax.plot(trial_nums, running_min, color=color, linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV-MAE (MW)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}\n")


def tune_xgboost(X_train, y_train, X_test, y_test, all_metrics, all_preds) -> dict:
    """Tune XGBoost with Optuna and append its test-set metrics/predictions."""
    print(f"Optuna tuning - XGBoost ({OPTUNA_TRIALS} trials x {OPTUNA_CV}-fold CV) ...\n")

    tscv_opt = TimeSeriesSplit(n_splits=OPTUNA_CV)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
            early_stopping_rounds=20,
            eval_metric="mae",
        )

        fold_maes = []
        for tr_idx, va_idx in tscv_opt.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            fold_maes.append(mean_absolute_error(y_va, model.predict(X_va)))
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="xgb_load_forecast",
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_cv_mae = study.best_value

    print(f"\nXGBoost Optuna finished - best CV-MAE: {best_cv_mae:.4f} MW")
    print("   Best hyperparameters:")
    for key, value in best_params.items():
        print(f"     {key:<22} = {value}")

    tuned_model = XGBRegressor(
        **best_params,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    tuned_model.fit(X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)

    metrics = compute_metrics("XGBoost (Tuned)", y_test, tuned_preds)
    all_metrics.append(metrics)
    all_preds["XGBoost (Tuned)"] = tuned_preds

    results_df = build_results_frame(all_metrics)
    best_model = best_model_name(results_df)

    print("\nXGBoost default vs Tuned - Test 2024")
    print(f"   {'Metric':<10}  {'Default':>10}  {'Tuned':>10}")
    print(f"   {'-' * 34}")
    for metric in ["MAE", "RMSE", "R2", "MAPE (%)"]:
        default_value = results_df.loc["XGBoost", metric]
        tuned_value = results_df.loc["XGBoost (Tuned)", metric]
        print(f"   {metric:<10}  {default_value:>10.4f}  {tuned_value:>10.4f}")

    valid_trials = [trial for trial in study.trials if trial.value is not None]
    trial_nums = [trial.number + 1 for trial in valid_trials]
    trial_maes = [trial.value for trial in valid_trials]
    running_min = pd.Series(trial_maes).cummin().tolist()

    _plot_optuna_convergence(
        trial_nums,
        trial_maes,
        running_min,
        "#E67E22",
        "Optuna Convergence - XGBoost",
        "plot_optuna_xgb_convergence.png",
    )

    return {
        "model": tuned_model,
        "preds": tuned_preds,
        "study": study,
        "trial_nums": trial_nums,
        "trial_maes": trial_maes,
        "running_min": running_min,
        "results_df": results_df,
        "best_model": best_model,
    }


def tune_lightgbm(X_train, y_train, X_test, y_test, all_metrics, all_preds) -> dict:
    """Tune LightGBM with Optuna and append its test-set metrics/predictions."""
    print(f"Optuna tuning - LightGBM ({LGBM_TRIALS} trials x {LGBM_CV}-fold CV) ...\n")

    tscv_lgbm = TimeSeriesSplit(n_splits=LGBM_CV)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 20, 300),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

        fold_maes = []
        for tr_idx, va_idx in tscv_lgbm.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            fold_maes.append(mean_absolute_error(y_va, model.predict(X_va)))
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="lgbm_load_forecast",
    )
    study.optimize(objective, n_trials=LGBM_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_cv_mae = study.best_value

    print(f"\nLightGBM Optuna finished - best CV-MAE: {best_cv_mae:.4f} MW")
    print("   Best hyperparameters:")
    for key, value in best_params.items():
        print(f"     {key:<22} = {value}")

    tuned_model = lgb.LGBMRegressor(
        **best_params,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    tuned_model.fit(X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)

    metrics = compute_metrics("LightGBM (Tuned)", y_test, tuned_preds)
    all_metrics.append(metrics)
    all_preds["LightGBM (Tuned)"] = tuned_preds

    results_df = build_results_frame(all_metrics)
    best_model = best_model_name(results_df)

    print("\nLightGBM default vs Tuned - Test 2024")
    print(f"   {'Metric':<10}  {'Default':>10}  {'Tuned':>10}")
    print(f"   {'-' * 34}")
    default_metrics = compute_metrics("LightGBM", y_test, all_preds["LightGBM"])
    for metric in ["MAE", "RMSE", "R2", "MAPE (%)"]:
        print(f"   {metric:<10}  {default_metrics[metric]:>10.4f}  {metrics[metric]:>10.4f}")

    valid_trials = [trial for trial in study.trials if trial.value is not None]
    trial_nums = [trial.number + 1 for trial in valid_trials]
    trial_maes = [trial.value for trial in valid_trials]
    running_min = pd.Series(trial_maes).cummin().tolist()

    _plot_optuna_convergence(
        trial_nums,
        trial_maes,
        running_min,
        "#1ABC9C",
        "Optuna Convergence - LightGBM",
        "plot_optuna_lgbm_convergence.png",
    )

    return {
        "model": tuned_model,
        "preds": tuned_preds,
        "study": study,
        "trial_nums": trial_nums,
        "trial_maes": trial_maes,
        "running_min": running_min,
        "results_df": results_df,
        "best_model": best_model,
    }

