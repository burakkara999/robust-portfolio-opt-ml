import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
layers = keras.layers

from sklearn.preprocessing import StandardScaler


def build_mlp(input_dim: int, output_dim: int, lr: float = 1e-3, hidden=(128, 64), dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for h in hidden:
        model.add(layers.Dense(h, activation="relu"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_dim, activation="linear"))  # multi-output regression

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def make_supervised_windows(returns_df: pd.DataFrame, lookback: int):
    """
    returns_df: (T, N) indexed by Date
    For each time t, X_t = returns[t-lookback:t] flattened, y_t = returns[t]
    """
    R = returns_df.to_numpy(dtype=float)
    T, N = R.shape

    X_list, y_list, idx_list = [], [], []
    for t in range(lookback, T):
        past = R[t - lookback:t, :]         # (lookback, N)
        if not np.isfinite(past).all():
            continue
        y = R[t, :]
        if not np.isfinite(y).all():
            continue

        X_list.append(past.reshape(-1))     # (lookback*N,)
        y_list.append(y)                    # (N,)
        idx_list.append(returns_df.index[t])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    idx = pd.DatetimeIndex(idx_list)
    return X, y, idx


def walk_forward_mlp_predictions(
    returns_df: pd.DataFrame,
    lookback: int = 60,
    train_periods: int = 260,     # number of training samples (rows) used each step
    start_date: str | None = None,
    end_date: str | None = None,
    hidden=(128, 64),
    dropout=0.1,
    lr=1e-3,
    epochs=50,
    batch_size=64,
    patience=5,
    seed=42,
    verbose=0,
):
    """
    Trains an MLP on a rolling window and predicts each next point (walk-forward).
    Produces a wide prediction DF aligned to dates.
    """
    tf.keras.utils.set_random_seed(seed)

    # Build supervised dataset
    X_all, y_all, idx_all = make_supervised_windows(returns_df, lookback=lookback)

    # Filter only prediction targets (keep full history for training)
    if start_date is not None:
        pred_mask = idx_all >= pd.to_datetime(start_date)
    else:
        pred_mask = np.ones(len(idx_all), dtype=bool)

    if end_date is not None:
        pred_mask &= (idx_all <= pd.to_datetime(end_date))

    if not pred_mask.any():
        raise ValueError("No prediction dates after filtering. Check start_date/end_date and lookback.")

    N_assets = returns_df.shape[1]
    input_dim = lookback * N_assets

    preds = []
    pred_dates = []

    # walk-forward: at each i, train on previous train_periods samples
    for i in range(len(idx_all)):
        if i < train_periods:
            # not enough supervised samples yet
            continue
        if not pred_mask[i]:
            # skip predictions outside requested date range
            continue

        X_train = X_all[i - train_periods:i]
        y_train = y_all[i - train_periods:i]
        X_test = X_all[i:i + 1]

        # scale X, y (fit on training only)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_s = x_scaler.fit_transform(X_train)
        X_test_s = x_scaler.transform(X_test)

        y_train_s = y_scaler.fit_transform(y_train)

        model = build_mlp(
            input_dim=input_dim,
            output_dim=N_assets,
            lr=lr,
            hidden=hidden,
            dropout=dropout
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            )
        ]

        model.fit(
            X_train_s, y_train_s,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )

        y_pred_s = model.predict(X_test_s, verbose=0)   # (1, N)
        y_pred = y_scaler.inverse_transform(y_pred_s)[0]

        preds.append(y_pred)
        pred_dates.append(idx_all[i])

    if len(pred_dates) == 0:
        raise ValueError(
            "No predictions were produced. Likely train_periods is too large relative to available data."
        )

    pred_df = pd.DataFrame(
        data=np.vstack(preds),
        index=pd.DatetimeIndex(pred_dates),
        columns=returns_df.columns
    ).sort_index()

    return pred_df


# -------------------------
# Example usage (your format)
# -------------------------
if __name__ == "__main__":
    markets_text = "crypto-commodities-bonds"

    returns_df = pd.read_csv(f"data/log_returns/log_returns_{markets_text}.csv")
    returns_df = returns_df.sort_values("Date").set_index("Date")
    returns_df.index = pd.to_datetime(returns_df.index)

    # MLP predictions (walk-forward)
    pred_df = walk_forward_mlp_predictions(
        returns_df=returns_df,
        lookback=50,
        train_periods=130,
        start_date="2024-06-03",
        end_date=None,
        hidden=(64, 32),
        dropout=0.2,
        lr=1e-3,
        epochs=50,
        batch_size=64,
        patience=5,
        verbose=0
    )

    # Residuals (true - predicted) aligned on prediction dates
    true_aligned = returns_df.loc[pred_df.index, pred_df.columns]
    resid_df = true_aligned - pred_df

    os.makedirs("data/prediction", exist_ok=True)
    pred_df.to_csv(f"data/prediction/MLP_{markets_text}_log_returns.csv")
    resid_df.to_csv(f"data/prediction/MLP_{markets_text}_residuals.csv")

    print("Saved:")
    print(" ", f"data/prediction/MLP_{markets_text}_log_returns.csv", pred_df.shape)
    print(" ", f"data/prediction/MLP_{markets_text}_residuals.csv", resid_df.shape)


## GRID tuning
# markets_text = "crypto-commodities-bonds"

# returns_df = pd.read_csv(f"data/log_returns/log_returns_{markets_text}.csv")
# returns_df = returns_df.sort_values("Date").set_index("Date")
# returns_df.index = pd.to_datetime(returns_df.index)

# param_grid = [
#     dict(lookback=20, train_periods=80,  hidden=(32, 16), dropout=0.0, lr=3e-4),
#     dict(lookback=30, train_periods=100, hidden=(32, 16), dropout=0.1, lr=1e-3),
#     dict(lookback=40, train_periods=100, hidden=(64, 32), dropout=0.1, lr=3e-4),
#     dict(lookback=50, train_periods=130, hidden=(64, 32), dropout=0.2, lr=1e-3),
# ]

# results = []
# for p in param_grid:
#     pred_df = walk_forward_mlp_predictions(
#         returns_df=returns_df,
#         start_date="2024-06-03",
#         end_date="2024-06-24",
#         epochs=50,
#         batch_size=64,
#         patience=5,
#         verbose=0,
#         **p
#     )
#     true_aligned = returns_df.loc[pred_df.index, pred_df.columns]
#     mse = ((true_aligned - pred_df) ** 2).mean().mean()
#     mae = (true_aligned - pred_df).abs().mean().mean()
#     results.append({**p, "mse": mse, "mae": mae})

# # sort by MAE (or MSE)
# results = sorted(results, key=lambda r: r["mae"])
# for r in results:
#     print(r)