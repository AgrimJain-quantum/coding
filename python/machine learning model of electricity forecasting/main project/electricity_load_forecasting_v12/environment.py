"""Runtime setup for plotting, logging, and warning noise."""

import warnings

import optuna
import seaborn as sns
import tensorflow as tf


def configure_runtime() -> None:
    """Apply the same global runtime settings used by the original script."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")

    sns.set_theme(style="darkgrid", palette="deep")
    sns.set_context("talk", font_scale=0.85)

    print(f"All libraries imported successfully. TensorFlow {tf.__version__}\n")

