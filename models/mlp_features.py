import numpy as np, pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


def fit_predict_mlp_Xfeat(train_windows, test_window, epochs=30, batch_size=256, lr=2e-4, seed=42):

    keras.utils.set_random_seed(seed)

    assets = test_window["y_ret"].index

    X_list, y_list = [], []
    for w in train_windows:
        Xw = w["X_feat"].reindex(index=assets)                 # (assets, F)
        yw = w["y_ret"].reindex(index=assets)                  # (assets,)
        X_list.append(Xw.to_numpy(np.float32))
        y_list.append(yw.to_numpy(np.float32))

    X_train = np.vstack(X_list)                                # (weeks*assets, F)
    y_train = np.concatenate(y_list)                            # (weeks*assets,)

    mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    X_train, y_train = X_train[mask], y_train[mask]

    X_test = test_window["X_feat"].reindex(index=assets).to_numpy(np.float32)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    return pd.Series(y_pred, index=assets)


def fit_predict_mlp_Xfeat_with_asset_onehot(
    train_windows, test_window, epochs=30, batch_size=256, lr=2e-4, seed=42
):
    keras.utils.set_random_seed(seed)

    assets = list(test_window["y_ret"].index)
    asset_to_id = {a:i for i,a in enumerate(assets)}
    A = len(assets)

    X_list, y_list, a_list = [], [], []
    for w in train_windows:
        Xw = w["X_feat"].reindex(index=assets)      # (A, F)
        yw = w["y_ret"].reindex(index=assets)       # (A,)
        X_list.append(Xw.to_numpy(np.float32))
        y_list.append(yw.to_numpy(np.float32))
        a_list.append(np.arange(A, dtype=np.int32)) # asset ids 0..A-1 (aligned with reindex)

    X_train = np.vstack(X_list)                    # (W*A, F)
    y_train = np.concatenate(y_list)               # (W*A,)
    a_train = np.concatenate(a_list)               # (W*A,)

    mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    X_train, y_train, a_train = X_train[mask], y_train[mask], a_train[mask]

    # one-hot
    a_train_oh = keras.utils.to_categorical(a_train, num_classes=A).astype(np.float32)
    X_train_aug = np.hstack([X_train, a_train_oh])  # (W*A, F + A)

    # test
    X_test = test_window["X_feat"].reindex(index=assets).to_numpy(np.float32)  # (A, F)
    a_test = np.arange(A, dtype=np.int32)
    a_test_oh = keras.utils.to_categorical(a_test, num_classes=A).astype(np.float32)
    X_test_aug = np.hstack([X_test, a_test_oh])     # (A, F + A)

    model = keras.Sequential([
        layers.Input(shape=(X_train_aug.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    model.fit(X_train_aug, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test_aug, verbose=0).reshape(-1)
    return pd.Series(y_pred, index=assets)