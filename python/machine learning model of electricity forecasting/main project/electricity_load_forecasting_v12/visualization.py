"""Plot generation for the v12 forecasting pipeline."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import COLORS, PLOT_PERIODS, ZOOM


def _save_and_show(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


def create_all_plots(
    dates_test,
    y_test,
    all_preds,
    results_df,
    rf_imp,
    xgb_imp,
    lgbm_imp,
    xgb_trial_nums,
    xgb_trial_maes,
    xgb_running_min,
    lgbm_trial_nums,
    lgbm_trial_maes,
    lgbm_running_min,
    w_xgb,
    w_lgbm,
    lstm_history,
) -> None:
    """Create and save all 13 plots from the original v12 workflow."""
    plot_dates = dates_test.values[-PLOT_PERIODS:]
    plot_actual = y_test.values[-PLOT_PERIODS:]

    # Plot 1: All models - Actual vs Predicted.
    rows = []
    for timestamp, actual in zip(plot_dates, plot_actual):
        rows.append({"datetime": timestamp, "Power Demand (MW)": actual, "Model": "Actual"})
    for name, preds in all_preds.items():
        for timestamp, pred in zip(plot_dates, preds[-PLOT_PERIODS:]):
            rows.append({"datetime": timestamp, "Power Demand (MW)": pred, "Model": name})

    plot1_df = pd.DataFrame(rows)
    plot1_df["datetime"] = pd.to_datetime(plot1_df["datetime"])
    color_map = {"Actual": "#2C3E50", **COLORS}

    fig, ax = plt.subplots(figsize=(22, 6))
    sns.lineplot(
        data=plot1_df,
        x="datetime",
        y="Power Demand (MW)",
        hue="Model",
        palette=color_map,
        linewidth=0.9,
        ax=ax,
    )
    for line in ax.get_lines():
        if line.get_label() == "Actual":
            line.set_linewidth(2.5)
            line.set_zorder(10)
    ax.set_title("Actual vs Predicted Power Demand - Last 7 Days (2024)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Power Demand (MW)")
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=20)
    _save_and_show("plot1_actual_vs_predicted.png")

    # Plot 2: Residuals.
    n_cols = 2
    n_rows = (len(all_preds) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4), sharex=True)
    axes = np.array(axes).flatten()

    for index, (name, preds) in enumerate(all_preds.items()):
        residuals = plot_actual - preds[-PLOT_PERIODS:]
        res_df = pd.DataFrame({"datetime": pd.to_datetime(plot_dates), "Residual": residuals})
        ax = axes[index]
        sns.lineplot(
            data=res_df,
            x="datetime",
            y="Residual",
            color=COLORS.get(name, "#AAB7B8"),
            linewidth=0.75,
            alpha=0.9,
            ax=ax,
        )
        ax.axhline(0, color="#2C3E50", linewidth=1.0, linestyle="--")
        ax.fill_between(
            res_df["datetime"],
            res_df["Residual"],
            0,
            where=(res_df["Residual"] > 0),
            alpha=0.15,
            color="#27AE60",
        )
        ax.fill_between(
            res_df["datetime"],
            res_df["Residual"],
            0,
            where=(res_df["Residual"] <= 0),
            alpha=0.15,
            color="#E74C3C",
        )
        ax.set_title(f"{name}  |  MAE = {results_df.loc[name, 'MAE']:.1f} MW", fontsize=9, fontweight="bold")
        ax.set_ylabel("Residual (MW)", fontsize=8)
        ax.set_xlabel("")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    for index in range(len(all_preds), len(axes)):
        axes[index].set_visible(False)

    plt.suptitle("Residuals (Actual - Predicted) - Last 7 Days", fontsize=14, fontweight="bold", y=1.01)
    _save_and_show("plot2_residuals.png")

    # Plot 3: Feature importances.
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, (title, imp, palette) in zip(
        axes,
        [
            ("Random Forest", rf_imp.head(15), "Greens_r"),
            ("XGBoost", xgb_imp.head(15), "Oranges_r"),
            ("LightGBM", lgbm_imp.head(15), "BuGn_r"),
        ],
    ):
        imp_df = imp.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        sns.barplot(
            data=imp_df,
            y="Feature",
            x="Importance",
            palette=palette,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
            orient="h",
        )
        for bar, value in zip(ax.patches, imp_df["Importance"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", ha="left", fontsize=8)
        ax.set_title(f"{title} - Top 15 Features", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("")

    plt.suptitle("Feature Importances: Random Forest vs XGBoost vs LightGBM", fontsize=14, fontweight="bold")
    _save_and_show("plot3_feature_importances.png")

    # Plot 4: Model comparison.
    metrics_to_plot = ["MAE", "RMSE", "MAPE (%)"]
    compare_long = results_df[metrics_to_plot].reset_index().melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value",
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, metric in zip(axes, metrics_to_plot):
        sub = compare_long[compare_long["Metric"] == metric].sort_values("Value")
        bar_colors = [COLORS.get(model, "#AAB7B8") for model in sub["Model"]]
        sns.barplot(
            data=sub,
            x="Model",
            y="Value",
            palette=bar_colors,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
        )
        for bar, value in zip(ax.patches, sub["Value"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{value:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        if ax.patches:
            ax.patches[0].set_edgecolor("gold")
            ax.patches[0].set_linewidth(3)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=40)

    plt.suptitle("Model Comparison - Test Set 2024 (Gold border = best)", fontsize=14, fontweight="bold")
    _save_and_show("plot4_model_comparison.png")

    # Plot 5: XGBoost vs RF vs LightGBM zoom.
    zoom_rows = []
    for timestamp, actual in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
        zoom_rows.append({"datetime": timestamp, "Power Demand (MW)": actual, "Model": "Actual"})
    for name in ["XGBoost", "Random Forest", "LightGBM"]:
        for timestamp, pred in zip(plot_dates[-ZOOM:], all_preds[name][-ZOOM:]):
            zoom_rows.append({"datetime": timestamp, "Power Demand (MW)": pred, "Model": name})

    zoom_df = pd.DataFrame(zoom_rows)
    zoom_df["datetime"] = pd.to_datetime(zoom_df["datetime"])
    zoom_pal = {
        "Actual": "#2C3E50",
        "XGBoost": "#E67E22",
        "Random Forest": "#27AE60",
        "LightGBM": "#1ABC9C",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=zoom_df, x="datetime", y="Power Demand (MW)", hue="Model", palette=zoom_pal, linewidth=1.2, ax=ax)
    for line in ax.get_lines():
        if line.get_label() == "Actual":
            line.set_linewidth(2.5)
            line.set_zorder(10)
    ax.set_title("XGBoost vs Random Forest vs LightGBM - Last 2 Days", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Power Demand (MW)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.xticks(rotation=20)
    _save_and_show("plot5_default_models_zoom.png")

    # Plot 6: Ensemble vs components.
    ens_rows = []
    for timestamp, actual in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
        ens_rows.append({"datetime": timestamp, "Power Demand (MW)": actual, "Model": "Actual"})
    for model_name in ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"]:
        for timestamp, pred in zip(plot_dates[-ZOOM:], all_preds[model_name][-ZOOM:]):
            ens_rows.append({"datetime": timestamp, "Power Demand (MW)": pred, "Model": model_name})

    ens_zoom_df = pd.DataFrame(ens_rows)
    ens_zoom_df["datetime"] = pd.to_datetime(ens_zoom_df["datetime"])
    ens_pal = {
        "Actual": "#2C3E50",
        "XGBoost (Tuned)": "#C0392B",
        "LightGBM (Tuned)": "#148F77",
        "Ensemble (XGB+LGBM)": "#6C3483",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=ens_zoom_df, x="datetime", y="Power Demand (MW)", hue="Model", palette=ens_pal, linewidth=1.2, ax=ax)
    for line in ax.get_lines():
        if line.get_label() == "Actual":
            line.set_linewidth(2.5)
            line.set_zorder(10)
        if line.get_label() == "Ensemble (XGB+LGBM)":
            line.set_linewidth(2.0)
    ax.set_title(f"Ensemble vs Components - Last 2 Days\nWeights -> XGB: {w_xgb:.3f}  |  LGBM: {w_lgbm:.3f}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Power Demand (MW)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.xticks(rotation=20)
    _save_and_show("plot6_ensemble_zoom.png")

    # Plot 7: Ensemble residual distribution.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model_name in zip(axes, ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"]):
        residuals = y_test.values - all_preds[model_name]
        color = ens_pal.get(model_name, "#AAB7B8")
        sns.histplot(residuals, bins=80, kde=True, color=color, edgecolor="white", linewidth=0.4, ax=ax)
        ax.axvline(0, color="#2C3E50", linewidth=1.2, linestyle="--")
        ax.axvline(np.mean(residuals), color="gold", linewidth=1.2, linestyle="--", label=f"Mean = {np.mean(residuals):.1f}")
        ax.set_title(f"{model_name}\nMAE = {results_df.loc[model_name, 'MAE']:.2f} MW", fontsize=10, fontweight="bold")
        ax.set_xlabel("Residual (MW)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    plt.suptitle("Residual Distribution - XGBoost (Tuned) vs LightGBM (Tuned) vs Ensemble", fontsize=13, fontweight="bold")
    _save_and_show("plot7_ensemble_residual_distribution.png")

    # Plot 8: XGBoost Optuna convergence.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(xgb_trial_nums, xgb_trial_maes, color="#95A5A6", s=18, alpha=0.6, label="Trial MAE")
    ax.plot(xgb_trial_nums, xgb_running_min, color="#E67E22", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV-MAE (MW)")
    ax.set_title("Optuna Convergence - XGBoost", fontsize=12, fontweight="bold")
    ax.legend()
    _save_and_show("plot8_optuna_xgb_convergence.png")

    # Plot 9: LightGBM Optuna convergence.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(lgbm_trial_nums, lgbm_trial_maes, color="#95A5A6", s=18, alpha=0.6, label="Trial MAE")
    ax.plot(lgbm_trial_nums, lgbm_running_min, color="#1ABC9C", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV-MAE (MW)")
    ax.set_title("Optuna Convergence - LightGBM", fontsize=12, fontweight="bold")
    ax.legend()
    _save_and_show("plot9_optuna_lgbm_convergence.png")

    # Plot 10: R2 comparison.
    r2_df = results_df[["R2"]].sort_values("R2", ascending=True).reset_index()
    bar_colors = [COLORS.get(model, "#AAB7B8") for model in r2_df["Model"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=r2_df, y="Model", x="R2", palette=bar_colors, edgecolor="white", linewidth=0.6, ax=ax, orient="h")
    for bar, value in zip(ax.patches, r2_df["R2"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_title("R2 Score - All Models (higher = better)", fontsize=13, fontweight="bold")
    ax.set_xlabel("R2 Score")
    ax.set_ylabel("")
    ax.axvline(1.0, color="#2C3E50", linewidth=0.8, linestyle="--", alpha=0.5)
    _save_and_show("plot10_r2_comparison.png")

    # Plot 11: Top 3 models, 1 day.
    zoom_1d = 288
    top3 = results_df.index[:3].tolist()
    top3_rows = []
    for timestamp, actual in zip(plot_dates[-zoom_1d:], plot_actual[-zoom_1d:]):
        top3_rows.append({"datetime": timestamp, "Power Demand (MW)": actual, "Model": "Actual"})
    for model_name in top3:
        for timestamp, pred in zip(plot_dates[-zoom_1d:], all_preds[model_name][-zoom_1d:]):
            top3_rows.append({"datetime": timestamp, "Power Demand (MW)": pred, "Model": model_name})

    top3_df = pd.DataFrame(top3_rows)
    top3_df["datetime"] = pd.to_datetime(top3_df["datetime"])
    top3_pal = {
        "Actual": "#2C3E50",
        top3[0]: COLORS.get(top3[0], "#E67E22"),
        top3[1]: COLORS.get(top3[1], "#C0392B"),
        top3[2]: COLORS.get(top3[2], "#148F77"),
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=top3_df, x="datetime", y="Power Demand (MW)", hue="Model", palette=top3_pal, linewidth=1.3, ax=ax)
    for line in ax.get_lines():
        if line.get_label() == "Actual":
            line.set_linewidth(2.5)
            line.set_zorder(10)
    ax.set_title(f"Top 3 Models vs Actual - Last 24 Hours\n1st: {top3[0]}  |  2nd: {top3[1]}  |  3rd: {top3[2]}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Power Demand (MW)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=20)
    _save_and_show("plot11_top3_models_1day.png")

    # Plot 12: LSTM vs ensemble.
    lstm_rows = []
    for timestamp, actual in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
        lstm_rows.append({"datetime": timestamp, "Power Demand (MW)": actual, "Model": "Actual"})
    for model_name in ["LSTM", "Ensemble (XGB+LGBM)", "XGBoost (Tuned)"]:
        for timestamp, pred in zip(plot_dates[-ZOOM:], all_preds[model_name][-ZOOM:]):
            lstm_rows.append({"datetime": timestamp, "Power Demand (MW)": pred, "Model": model_name})

    lstm_zoom_df = pd.DataFrame(lstm_rows)
    lstm_zoom_df["datetime"] = pd.to_datetime(lstm_zoom_df["datetime"])
    lstm_pal = {
        "Actual": "#2C3E50",
        "LSTM": "#D4145A",
        "Ensemble (XGB+LGBM)": "#6C3483",
        "XGBoost (Tuned)": "#C0392B",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=lstm_zoom_df, x="datetime", y="Power Demand (MW)", hue="Model", palette=lstm_pal, linewidth=1.2, ax=ax)
    for line in ax.get_lines():
        if line.get_label() == "Actual":
            line.set_linewidth(2.5)
            line.set_zorder(10)
        if line.get_label() == "LSTM":
            line.set_linewidth(2.0)
            line.set_zorder(9)
    ax.set_title(
        f"LSTM vs Ensemble vs XGBoost (Tuned) - Last 2 Days\n"
        f"LSTM MAE={results_df.loc['LSTM', 'MAE']:.2f}  |  "
        f"Ensemble MAE={results_df.loc['Ensemble (XGB+LGBM)', 'MAE']:.2f}  |  "
        f"XGB(T) MAE={results_df.loc['XGBoost (Tuned)', 'MAE']:.2f}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Power Demand (MW)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.xticks(rotation=20)
    _save_and_show("plot12_lstm_vs_ensemble_zoom.png")

    # Plot 13: LSTM training history.
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs_ran = range(1, len(lstm_history.history["loss"]) + 1)
    ax.plot(epochs_ran, lstm_history.history["loss"], color="#D4145A", linewidth=2, label="Train Loss (MSE)")
    ax.plot(epochs_ran, lstm_history.history["val_loss"], color="#2C3E50", linewidth=2, linestyle="--", label="Val Loss (MSE)")
    best_epoch = int(np.argmin(lstm_history.history["val_loss"])) + 1
    ax.axvline(best_epoch, color="gold", linewidth=1.5, linestyle=":", label=f"Best epoch = {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.set_title("LSTM Training History - Train vs Validation Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    _save_and_show("plot13_lstm_training_history.png")

