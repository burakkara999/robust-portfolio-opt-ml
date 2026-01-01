import numpy as np, pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


def fit_predict_mlp_per_asset_Xfeat(train_windows, test_window, epochs=30, batch_size=32, lr=2e-4, seed=42, min_samples=10):
    
    keras.utils.set_random_seed(seed)

    assets = test_window["y_ret"].index
    X_te = test_window["X_feat"].reindex(index=assets).to_numpy(np.float32)

    preds = pd.Series(index=assets, dtype=float)

    for i, a in enumerate(assets):
        X_rows, y_rows = [], []
        for w in train_windows:
            x = w["X_feat"].reindex(index=[a]).to_numpy(np.float32).reshape(-1)
            y = w["y_ret"].get(a, np.nan)
            if np.isfinite(y) and np.isfinite(x).all():
                X_rows.append(x)
                y_rows.append(float(y))

        if len(y_rows) < min_samples:
            preds.loc[a] = float(np.mean(y_rows)) if len(y_rows) else 0.0
            continue

        X_tr = np.vstack(X_rows)
        y_tr = np.array(y_rows, dtype=np.float32)

        model = keras.Sequential([
            layers.Input(shape=(X_tr.shape[1],)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        preds.loc[a] = float(model.predict(X_te[i].reshape(1,-1), verbose=0).reshape(-1)[0])

    return preds