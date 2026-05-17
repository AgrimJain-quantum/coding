"""LSTM sequence model for the v12 forecasting pipeline."""

import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .config import BATCH_SIZE, FEATURES, LOOKBACK, MAX_EPOCHS, SPLIT_DATE, TARGET
from .metrics import best_model_name, build_results_frame, compute_metrics


def build_sequences(data: np.ndarray, lookback: int, target_col_idx: int):
    """Return sequence-to-one samples for an already-scaled array."""
    X_seqs, y_seqs = [], []
    for index in range(lookback, len(data)):
        X_seqs.append(data[index - lookback : index, :])
        y_seqs.append(data[index, target_col_idx])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


def train_lstm(df: pd.DataFrame, y_test, all_metrics, all_preds) -> dict:
    """Train, predict, and evaluate the LSTM model."""
    print("\nTraining LSTM ...")

    lstm_feature_cols = FEATURES + [TARGET]
    lstm_scaler = MinMaxScaler(feature_range=(0, 1))

    train_idx = df.index[df["datetime"] < SPLIT_DATE]

    lstm_scaler.fit(df.loc[train_idx, lstm_feature_cols])
    scaled_all = lstm_scaler.transform(df[lstm_feature_cols])

    scaled_df = pd.DataFrame(scaled_all, columns=lstm_feature_cols, index=df.index)
    target_col_idx = lstm_feature_cols.index(TARGET)

    scaled_train = scaled_df.loc[train_idx].values
    X_lstm_train, y_lstm_train = build_sequences(scaled_train, LOOKBACK, target_col_idx)

    boundary = train_idx[-1]
    boundary_pos = df.index.get_loc(boundary)
    context_start = boundary_pos - LOOKBACK + 1
    scaled_ctx_and_test = scaled_df.values[context_start:]
    X_lstm_test, _ = build_sequences(scaled_ctx_and_test, LOOKBACK, target_col_idx)

    assert len(X_lstm_test) == len(y_test), (
        f"LSTM test length mismatch: {len(X_lstm_test)} vs {len(y_test)}"
    )

    n_features = X_lstm_train.shape[2]
    print(f"   Train sequences : {X_lstm_train.shape}")
    print(f"   Test sequences  : {X_lstm_test.shape}")
    print(f"   Features        : {n_features}  |  Lookback: {LOOKBACK} steps")

    tf.random.set_seed(42)
    lstm_model = Sequential(
        [
            Input(shape=(LOOKBACK, n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="LSTM_LoadForecast",
    )

    lstm_model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    lstm_model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = lstm_model.fit(
        X_lstm_train,
        y_lstm_train,
        validation_split=0.1,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n   Stopped at epoch {len(history.history['loss'])}")

    lstm_scaled_preds = lstm_model.predict(
        X_lstm_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).flatten()

    dummy = np.zeros((len(lstm_scaled_preds), len(lstm_feature_cols)), dtype=np.float32)
    dummy[:, target_col_idx] = lstm_scaled_preds
    lstm_preds_mw = lstm_scaler.inverse_transform(dummy)[:, target_col_idx]

    metrics = compute_metrics("LSTM", y_test, lstm_preds_mw)
    all_metrics.append(metrics)
    all_preds["LSTM"] = lstm_preds_mw

    results_df = build_results_frame(all_metrics)
    best_model = best_model_name(results_df)

    print("\nLSTM - Test 2024")
    print(f"   MAE  = {metrics['MAE']:.2f} MW")
    print(f"   RMSE = {metrics['RMSE']:.2f} MW")
    print(f"   R2   = {metrics['R2']:.4f}")
    print(f"   MAPE = {metrics['MAPE (%)']:.2f}%\n")

    gc.collect()
    tf.keras.backend.clear_session()

    return {
        "history": history,
        "preds": lstm_preds_mw,
        "results_df": results_df,
        "best_model": best_model,
    }

