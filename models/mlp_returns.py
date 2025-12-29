import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from dataset_builders import X_from_past_returns

def fit_predict(train_windows, test_window, lookback_days=25, epochs=20, batch_size=256, lr=2e-4):
    # Build train X/y by stacking assets across windows
    Xs, ys = [], []
    common = None
    for w in train_windows:
        assets_w = w["y_ret"].index
        common = assets_w if common is None else common.intersection(assets_w)
    common = common.sort_values()

    for w in train_windows:
        X = X_from_past_returns(w, lookback_days=lookback_days)  # (assets, d)
        y = w["y_ret"].loc[common].to_numpy(dtype=np.float32)    # (assets,)
        pos = w["y_ret"].index.get_indexer(common)
        Xs.append(X[pos, :])
        ys.append(y.reshape(-1, 1))

    X_train = np.vstack(Xs).astype(np.float32)
    y_train = np.vstack(ys).astype(np.float32).ravel()

    # Normalize using train only
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sigma

    # Test X (assets, d) in same asset order as test_window
    X_test = X_from_past_returns(test_window, lookback_days=lookback_days)
    assets_test = test_window["y_ret"].index
    X_test = (X_test - mu) / sigma

    # MLP
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    return pd.Series(y_pred, index=assets_test)



def fit_predict_per_asset(train_windows, test_window, lookback_days=25, epochs=20, batch_size=32, lr=2e-4):
    assets = test_window["y_ret"].index
    y_pred = pd.Series(index=assets, dtype=float)

    # Build per-asset training series: (weeks, d) -> predict next-week return
    for a in assets:
        # collect training samples for this asset from each train window
        X_list = []
        y_list = []
        ok = True
        for w in train_windows:
            if a not in w["y_ret"].index:
                ok = False
                break
            Xw = X_from_past_returns(w, lookback_days=lookback_days)   # (assets, d)
            pos = w["y_ret"].index.get_loc(a)
            X_list.append(Xw[pos, :])
            y_list.append(float(w["y_ret"].loc[a]))
        if not ok:
            continue

        X_tr = np.asarray(X_list, dtype=np.float32)  # (52, d)
        y_tr = np.asarray(y_list, dtype=np.float32)  # (52,)

        # normalize per-asset using its own history (train only)
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sigma

        # test x for that asset
        X_te_all = X_from_past_returns(test_window, lookback_days=lookback_days)
        pos_te = test_window["y_ret"].index.get_loc(a)
        x_te = X_te_all[pos_te, :].astype(np.float32)
        x_te = (x_te - mu) / sigma
        x_te = x_te.reshape(1, -1)

        # tiny MLP (keep simple to avoid overfitting with 52 points)
        model = keras.Sequential([
            layers.Input(shape=(X_tr.shape[1],)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred.loc[a] = float(model.predict(x_te, verbose=0).reshape(-1)[0])

    # Fill any missing assets if needed (e.g., with 0 or cross-sectional mean)
    y_pred = y_pred.fillna(0.0)
    return y_pred