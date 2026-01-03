import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from dataset_builders import X_from_past_returns


def fit_predict_mlp_per_asset(
    train_windows,
    test_window,
    lookback_days=25,
    epochs=30,
    batch_size=32,
    lr=2e-4
):
    """
    Per-asset model:
      For each asset a:
        - Build (train_weeks, d) samples from X_from_past_returns(window)
        - Fit a tiny MLP on that asset only
        - Predict next-week return for that asset in test_window

    Returns:
      pd.Series(index=test_window["y_ret"].index, values=y_pred)
    """
    assets = test_window["y_ret"].index
    y_pred = pd.Series(index=assets, dtype=float)

    # Precompute test X once (assets, d)
    X_test_all = X_from_past_returns(test_window, lookback_days=lookback_days).astype(np.float32)

    for a in assets:
        # Collect training samples for this asset across train windows
        X_list = []
        y_list = []
        ok = True

        for w in train_windows:
            if a not in w["y_ret"].index:
                ok = False
                break

            Xw_all = X_from_past_returns(w, lookback_days=lookback_days).astype(np.float32)  # (assets, d)
            pos = w["y_ret"].index.get_loc(a)
            X_list.append(Xw_all[pos, :])
            y_list.append(float(w["y_ret"].loc[a]))

        if not ok or len(X_list) < 5:
            continue

        X_tr = np.asarray(X_list, dtype=np.float32)  # (W, d)
        y_tr = np.asarray(y_list, dtype=np.float32)  # (W,)
        X_tr.shape
        print("DEBUG on mlp per asset")
        # Normalize per-asset (train only)
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sigma

        # Test row for this asset
        pos_te = test_window["y_ret"].index.get_loc(a)
        x_te = X_test_all[pos_te, :]
        x_te = ((x_te - mu) / sigma).reshape(1, -1)

        # Tiny model (52 points => keep it small)
        model = keras.Sequential(
            [
                layers.Input(shape=(X_tr.shape[1],)),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred.loc[a] = float(model.predict(x_te, verbose=0).reshape(-1)[0])

    return y_pred.fillna(0.0)