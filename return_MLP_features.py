
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from stock_data_module import read_close_prices_all_merged, read_close_prices
from features import make_feature_windows



###
def _build_xy_from_feature_windows(rolling_feature_windows):
    rows = []
    for t_idx, w in enumerate(rolling_feature_windows):
        X = w["X_feat"]      # DataFrame: assets x features
        y = w["y_ret"]  # Series: assets

        y = y.reindex(X.index)  # align

        # make long-form samples: one row per (window, asset)
        df = X.copy()
        df["y"] = y.values
        df["t_idx"] = t_idx
        df["asset"] = df.index

        # drop non-finite
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(df)

    all_df = pd.concat(rows, axis=0)
    feature_cols = [c for c in all_df.columns if c not in ["y", "t_idx", "asset"]]

    X_all = all_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = all_df["y"].to_numpy(dtype=np.float32)
    t_all = all_df["t_idx"].to_numpy(dtype=np.int32)
    asset_all = all_df["asset"].to_numpy()

    return X_all, y_all, t_all, asset_all, feature_cols


def _time_split_and_scale(X, y, t, train_frac=0.6, val_frac=0.2):
    T = int(t.max() + 1)          # number of windows
    train_end = int(T * train_frac)
    val_end   = int(T * (train_frac + val_frac))

    tr = t < train_end
    va = (t >= train_end) & (t < val_end)
    te = t >= val_end

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8

    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std
    X_te = (X_te - mean) / std

    return X_tr, y_tr, X_va, y_va, X_te, y_te, mean, std, train_end, val_end


def _make_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)  # regression output
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.Huber(delta=1e-3),  # robust vs outliers
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def train_mlp(rolling_feature_windows, train_frac=0.6, val_frac=0.2,
                   batch_size=512, epochs=50):
    X, y, t, asset, feature_cols = _build_xy_from_feature_windows(rolling_feature_windows)

    X_tr, y_tr, X_va, y_va, X_te, y_te, mean, std, train_end, val_end = \
        _time_split_and_scale(X, y, t, train_frac=train_frac, val_frac=val_frac)

    model = _make_mlp(X_tr.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, mean, std, feature_cols, train_end, val_end


def evaluate_mlp_on_test_windows(rolling_feature_windows, model, x_mean, x_std, test_start_t):
    X, y, t, asset, feature_cols = _build_xy_from_feature_windows(rolling_feature_windows)
    Xs = (X - x_mean) / x_std
    yhat = model.predict(Xs, verbose=0).reshape(-1)

    test_mask = (t >= test_start_t)

    y_test = y[test_mask]
    yhat_test = yhat[test_mask]
    t_test = t[test_mask]
    asset_test = asset[test_mask]

    mse = float(np.mean((yhat_test - y_test) ** 2))
    mae = float(np.mean(np.abs(yhat_test - y_test)))

    df = pd.DataFrame({"t": t_test, "asset": asset_test, "y": y_test, "yhat": yhat_test})

    ic_by_week  = df.groupby("t").apply(lambda g: g["y"].corr(g["yhat"])).dropna()
    ric_by_week = df.groupby("t").apply(lambda g: g["y"].rank().corr(g["yhat"].rank())).dropna()

    results = {
        "test_mse": mse,
        "test_mae": mae,
        "mean_IC": float(ic_by_week.mean()) if len(ic_by_week) else np.nan,
        "mean_RankIC": float(ric_by_week.mean()) if len(ric_by_week) else np.nan,
        "n_test_windows": int(df["t"].nunique()),
        "n_test_samples": int(len(df)),
    }
    return results, df




# markets = ['bist100', 'dow30', 'commodities', 'bonds', 'funds_mini']
# markets = ['bist100']
markets = ['dow30']

results_by_market = {}
models_by_market = {}

for market in markets:
    print(f"\n===== Training model for {market} =====")

    tickers, close_df = read_close_prices_all_merged([market])  ## 980days x N assets
    
    ##191 weeks = 980/5 - lookback*5
    rolling_feature_windows = make_feature_windows(close_prices=close_df, lookback=5) ## 191weeks --> X: N assets x 9 features

    print(rolling_feature_windows[0].keys())
    print(rolling_feature_windows[0]['X_feat'].shape)
    
    # X, y, t, asset, feature_cols = _build_xy_from_feature_windows(rolling_feature_windows) ## X: (191*N, 9), y: (191*N, )
    # print(X.shape, y.shape)

    model, x_mean, x_std, feature_cols, train_end, val_end = train_mlp(
        rolling_feature_windows,
        train_frac=0.6,
        val_frac=0.2
    )

    results, test_df = evaluate_mlp_on_test_windows(
        rolling_feature_windows,
        model,
        x_mean,
        x_std,
        test_start_t=val_end
    )

    print(results)

    models_by_market[market] = (model, x_mean, x_std, feature_cols)
    results_by_market[market] = results

