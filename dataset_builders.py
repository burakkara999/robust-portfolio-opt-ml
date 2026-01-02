import numpy as np
import pandas as pd

def get_assets(w) -> pd.Index:
    # make_feature_windows() already aligns and sorts assets
    return w["X_feat"].index

def X_from_features(w) -> np.ndarray:
    # (assets, F)
    return w["X_feat"].to_numpy(dtype=np.float32)

def X_from_past_returns(w, lookback_days=25) -> np.ndarray:
    # w["past_returns"]: DataFrame (days, assets)
    pr = w["past_returns"].iloc[-lookback_days:, :]  # (days, assets)
    return pr.to_numpy(dtype=np.float32).T           # (assets, days)

def y_from_next_week_return(w) -> np.ndarray:
    # w["y_ret"]: Series (assets,)
    return w["y_ret"].to_numpy(dtype=np.float32)


def X_from_weekly_return_lags(train_windows, test_window, n_lags=12) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Build pooled dataset from weekly returns:
      For each week i in train_windows (i >= n_lags):
        X(i) = [y_ret(i-n_lags), ..., y_ret(i-1)]
        y(i) = y_ret(i)
    And for the test week:
        X_test = [last n_lags weeks from train_windows]
    Returns: X_train, y_train, X_test, assets
    """
    assets = test_window["y_ret"].index

    # weekly return matrix over training windows: (W, A)
    Y = pd.DataFrame([w["y_ret"].reindex(assets) for w in train_windows]).astype(float)

    W = len(train_windows)
    if W <= n_lags:
        raise ValueError(f"Need > n_lags windows. Have W={W}, n_lags={n_lags}")

    X_list, y_list = [], []

    # training samples
    for i in range(n_lags, W):
        # features are previous n_lags weekly returns
        X_i = Y.iloc[i - n_lags:i].to_numpy(dtype=np.float32).T   # (A, n_lags)
        y_i = Y.iloc[i].to_numpy(dtype=np.float32)               # (A,)
        X_list.append(X_i)
        y_list.append(y_i)

    X_train = np.vstack(X_list)          # (A*(W-n_lags), n_lags)
    y_train = np.concatenate(y_list)     # (A*(W-n_lags),)

    # test features = last n_lags weekly returns before test window
    X_test = Y.iloc[W - n_lags:W].to_numpy(dtype=np.float32).T   # (A, n_lags)

    # drop NaNs if any
    mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    X_train, y_train = X_train[mask], y_train[mask]
    X_test = np.nan_to_num(X_test, nan=0.0)

    return X_train, y_train, X_test, assets