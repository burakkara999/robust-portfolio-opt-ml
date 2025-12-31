# models/rolling_mean.py

import numpy as np
import pandas as pd


def fit_predict_rolling_mean(train_windows, test_window, fillna_value=0.0):
    """
    Rolling sample-mean benchmark.

    For each asset a:
      y_hat[a] = mean_{w in train_windows} ( w["y_ret"][a] )

    Returns:
      pd.Series indexed by test_window["y_ret"].index
    """
    assets = test_window["y_ret"].index

    # Collect y_ret for each training week into a (weeks x assets) DataFrame
    rows = []
    for w in train_windows:
        s = w["y_ret"].reindex(assets)  # align to test assets
        rows.append(s)

    Y = pd.DataFrame(rows)  # shape: (train_weeks, n_assets)

    y_pred = Y.mean(axis=0).astype(float)  # sample mean per asset
    return y_pred.fillna(fillna_value)