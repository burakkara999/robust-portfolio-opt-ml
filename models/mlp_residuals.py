import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def fit_predict_residual_mlp_Xfeat(
    train_windows,
    test_window,
    epochs=50,
    batch_size=256,
    lr=2e-4,
    seed=42,
    residual_scale=1.0,      # shrink residuals (0.25/0.5/1.0)
    l2=1e-4,
    dropout=0.1,
    add_mu_to_X=False,       # optional: append sample-mean as extra feature
    fillna_value=0.0,
):
    """
    Predict next-week returns as:
        y_hat = mu_sample_mean(train) + residual_scale * MLP(X_feat)->residual

    Train target:
        r = y - mu_sample_mean(train)
    """
    tf.keras.utils.set_random_seed(seed)

    assets = test_window["y_ret"].index

    # --- sample mean baseline mu (per asset) ---
    Y = pd.DataFrame([w["y_ret"].reindex(assets) for w in train_windows])  # (W, A)
    mu = Y.mean(axis=0).astype(float).fillna(fillna_value)                # (A,)

    # --- build pooled training set: (window, asset) rows ---
    X_list, r_list = [], []
    for w in train_windows:
        Xw = w["X_feat"].reindex(index=assets)                    # (A, F)
        yw = w["y_ret"].reindex(index=assets).astype(float)       # (A,)
        rw = (yw - mu)                                            # (A,)

        X_arr = Xw.to_numpy(np.float32)
        r_arr = rw.to_numpy(np.float32)

        if add_mu_to_X:
            mu_col = mu.to_numpy(np.float32).reshape(-1, 1)       # (A, 1)
            X_arr = np.hstack([X_arr, mu_col])                    # (A, F+1)

        X_list.append(X_arr)
        r_list.append(r_arr)

    X_train = np.vstack(X_list)                                   # (W*A, F or F+1)
    r_train = np.concatenate(r_list)                              # (W*A,)

    mask = np.isfinite(r_train) & np.isfinite(X_train).all(axis=1)
    X_train, r_train = X_train[mask], r_train[mask]

    # fallback if too little data
    if X_train.shape[0] < 50:
        return mu

    # --- test features ---
    X_test = test_window["X_feat"].reindex(index=assets).to_numpy(np.float32)  # (A, F)
    if add_mu_to_X:
        X_test = np.hstack([X_test, mu.to_numpy(np.float32).reshape(-1, 1)])

    X_test = np.nan_to_num(X_test, nan=0.0)

    # --- model with train-only normalization ---
    norm = layers.Normalization()
    norm.adapt(X_train)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        norm,
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)),
        layers.Dense(1),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    cb = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    ]

    model.fit(X_train, r_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)

    r_pred = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1).astype(float)

    y_pred = mu.to_numpy(float) + residual_scale * r_pred
    return pd.Series(y_pred, index=assets).fillna(fillna_value)