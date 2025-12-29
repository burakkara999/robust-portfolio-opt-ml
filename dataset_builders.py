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