import numpy as np
import pandas as pd

from asset_data_module import read_close_prices_all_merged
from features import make_feature_windows
from portfolio_models import solve_markowitz_robust


def stack_train_xy(train_windows, X_fn, y_fn):
    """
    Train samples are (window, asset).
    Returns:
      X_train: (train_windows * A, d)
      y_train: (train_windows * A,)
      assets:  Index of assets used (must be consistent across windows)
    """
    # enforce consistent universe via intersection across training windows
    common = None
    for w in train_windows:
        assets_w = w["y_ret"].index
        common = assets_w if common is None else common.intersection(assets_w)
    common = common.sort_values()

    Xs, ys = [], []
    for w in train_windows:
        # slice to common assets
        X = X_fn(w)
        y = y_fn(w)
        assets_w = w["y_ret"].index

        # build a selector from w's asset order to common order
        pos = assets_w.get_indexer(common)
        Xs.append(X[pos, :])
        ys.append(y[pos].reshape(-1, 1))

    X_train = np.vstack(Xs).astype(np.float32)
    y_train = np.vstack(ys).astype(np.float32).ravel()
    return X_train, y_train, common


def standardize_fit(X_train):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


def weekly_return_cov_from_train_windows(train_windows, assets):
    """
    Σ_r(t): covariance of weekly realized returns.
    Use y_ret from each training window => matrix shape (W, N).
    """
    R = np.vstack([w["y_ret"].loc[assets].to_numpy(dtype=float) for w in train_windows])  # (W, N)
    Sigma = np.cov(R, rowvar=False)
    return Sigma


def error_cov_from_history(errors_by_week, assets, last_m=20, ridge=1e-8):
    """
    Λ(t): covariance of weekly prediction errors.
    errors_by_week: list[pd.Series(index=assets, values=err)]
    Uses last_m available error vectors (must be OOS realized).
    """
    if len(errors_by_week) < last_m:
        return None

    E_list = []
    for e in errors_by_week[-last_m:]:
        # align to assets (intersection already enforced in runner)
        E_list.append(e.loc[assets].to_numpy(dtype=float))
    E = np.vstack(E_list)  # (M, N)

    Lambda = np.cov(E, rowvar=False)
    # tiny ridge helps numerical stability
    Lambda = Lambda + ridge * np.eye(Lambda.shape[0])
    return Lambda


def run_walk_forward(
    markets: list[str],
    model_fit_predict_fn,   # (train_windows, test_window) -> pd.Series(index=assets, values=y_pred)
    X_fn,
    y_fn,
    lookback_weeks=5,
    horizon_weeks=1,
    train_weeks=52,
    error_weeks=20,
    start_date="2022-01-01",
    end_date="2025-11-30",
    kappa=0.5,
    delta=0.5,
    out_preds_csv=None,
    out_ports_csv=None,
    out_ports_summary_csv=None,
):
    # -------------------------
    # Load data + make windows
    # -------------------------
    tickers_dict, close_df = read_close_prices_all_merged(markets, after_date=start_date)
    close_df = close_df.loc[:end_date]

    rolling = make_feature_windows(
        close_prices=close_df,
        lookback=lookback_weeks,
        horizon=horizon_weeks,
    )

    # -------------------------
    # Walk-forward
    # -------------------------
    preds_rows = []
    ports_rows = []
    port_summary_rows = [] 
    
    errors_by_week = []  # list of pd.Series: err_t over assets for each realized test week

    robust_start_k = train_weeks + error_weeks  # 72 if 52+20

    for k in range(train_weeks, len(rolling)):
        train_windows = rolling[k - train_weeks : k]
        test_window = rolling[k]

        # universe for THIS step: intersection(train universe, test universe)
        # (keeps alignment safe if assets drop in/out)
        common_assets = test_window["y_ret"].index
        for w in train_windows:
            common_assets = common_assets.intersection(w["y_ret"].index)
        common_assets = common_assets.sort_values()

        ## if universe too small, skip
        if len(common_assets) < 2:
            continue

        ## --- model prediction (DATA MODEL CALLS) ---
        y_pred = model_fit_predict_fn(train_windows, test_window)  # Series indexed by assets
        y_true = test_window["y_ret"]


        y_pred = y_pred.loc[common_assets]
        y_true = y_true.loc[common_assets]
        err = (y_true - y_pred)

        # store prediction rows
        for a in common_assets:
            preds_rows.append({
                "t0": test_window["t0"],
                "t1": test_window["t1"],
                "asset": a,
                "y_true": float(y_true.loc[a]),
                "y_pred": float(y_pred.loc[a]),
                "err": float(err.loc[a]),
            })

        # add realized error vector to history (for future Λ)
        errors_by_week.append(err)

        # --- risk covariance Σ from last 52 weeks (train windows) ---
        Sigma = weekly_return_cov_from_train_windows(train_windows, common_assets)

        # --- error covariance Λ from last 20 realized OOS error vectors ---
        Lambda = error_cov_from_history(errors_by_week, common_assets, last_m=error_weeks)

        # --- portfolio optimization ---
        if (k >= robust_start_k) and (Lambda is not None):
            w_opt, obj = solve_markowitz_robust(
                assets=list(common_assets),
                expected_returns=y_pred.to_numpy(dtype=float),
                Sigma=Sigma,
                Lambda=Lambda,
                kappa=kappa,
                delta=delta,
            )
            is_robust = True
        else:
            # warm-up period: run "non-robust" by setting Lambda ~ 0
            # (so the uncertainty penalty disappears)
            Lambda0 = 1e-12 * np.eye(len(common_assets))
            w_opt, obj = solve_markowitz_robust(
                assets=list(common_assets),
                expected_returns=y_pred.to_numpy(dtype=float),
                Sigma=Sigma,
                Lambda=Lambda0,
                kappa=0.0,          # turn off uncertainty penalty during warm-up
                delta=delta,
            )
            is_robust = False


        mu_hat = y_pred.to_numpy(dtype=float)  # (N,)
        r_real = y_true.to_numpy(dtype=float)  # (N,)
        expected_portfolio_return = float(np.dot(w_opt, mu_hat))
        realized_portfolio_return = float(np.dot(w_opt, r_real))
        
        ## portfolio summary 
        port_summary_rows.append({
            "t0": test_window["t0"],
            "t1": test_window["t1"],
            "expected_portfolio_return": expected_portfolio_return,
            "realized_portfolio_return": realized_portfolio_return,
            "objective": float(obj),
            "is_robust": int(is_robust),
            "n_assets": int(len(common_assets)),
            "sum_w": float(w_opt.sum()),
        })
        ## portfolio details
        for i, a in enumerate(common_assets):
            ports_rows.append({
                "t0": test_window["t0"],
                "t1": test_window["t1"],
                "asset": a,
                "weight": float(w_opt[i]),
                "expected_portfolio_return": expected_portfolio_return,
                "realized_portfolio_return": realized_portfolio_return,
                "objective": float(obj),
                "is_robust": int(is_robust),
            })

    preds_df = pd.DataFrame(preds_rows)
    ports_df = pd.DataFrame(ports_rows)
    port_summary_df = pd.DataFrame(port_summary_rows)

    if out_preds_csv:
        preds_df.to_csv(out_preds_csv, index=False)
    if out_ports_csv:
        ports_df.to_csv(out_ports_csv, index=False)
    if out_ports_summary_csv:
        port_summary_df.to_csv("outputs/portfolio_summary.csv", index=False)

    return preds_df, ports_df