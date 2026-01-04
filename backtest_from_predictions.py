
"""
Backtest robust Markowitz portfolio optimization using PRECOMPUTED prediction tables.

Inputs (wide tables; index=period, columns=assets):
  - {model_name}_{markets}_expected_returns.csv
  - {model_name}_{markets}_true_returns.csv
  - {model_name}_{markets}_errors.csv   (optional; if missing we compute true - expected)

We do NOT call any prediction model here. We only read the saved tables and run portfolio optimization.

Typical usage:
  python backtest_from_predictions.py \
      --markets dow30 \
      --model-name MLP \
      --year 2025 \
      --pred-dir data/prediction \
      --train-weeks 52 \
      --error-weeks 20 \
      --warmup-periods 20 \
      --kappa 0.5 --delta 0.5 \
      --out-dir data/backtests

Notes:
- To estimate Sigma/Lambda at early periods, this script uses *available* history up to t-1.
  If your prediction tables contain only 2025 rows, early Sigma/Lambda will be based on few samples.
  (If you want richer covariance estimation, export predictions for 2024 too, and keep those rows in the CSV.)
"""

import os
import numpy as np
import pandas as pd

from portfolio_models import solve_markowitz_robust


def _read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # try parse index as datetime if possible
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df.columns = df.columns.astype(str)
    return df.sort_index()


def _safe_cov(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    """
    mat: (T, N) matrix where rows are time and cols are assets
    Returns: (N, N) covariance with tiny ridge for numerical stability.
    """
    if mat.ndim != 2:
        raise ValueError("mat must be 2D")
    if mat.shape[0] < 2:
        return None
    Sigma = np.cov(mat, rowvar=False)
    Sigma = Sigma + eps * np.eye(Sigma.shape[0])
    return Sigma


def _align_universe_for_period(
    expected_row: pd.Series,
    true_row: pd.Series,
    hist_true: pd.DataFrame | None,
    hist_err: pd.DataFrame | None,
    min_assets: int = 2,
) -> pd.Index | None:
    """
    Decide asset universe at a given period:
      - must have finite expected + true today
      - must have enough history for Sigma/Lambda estimation (drop assets with NA in history)
    """
    assets = expected_row.index

    true_aligned = true_row.reindex(assets)
    mask_today = np.isfinite(expected_row.values) & np.isfinite(true_aligned.values)
    assets_today = assets[mask_today]

    if hist_true is not None and not hist_true.empty:
        ok_hist_true = hist_true[assets_today].notna().all(axis=0)
        assets_today = assets_today[ok_hist_true.values]

    if hist_err is not None and not hist_err.empty:
        ok_hist_err = hist_err[assets_today].notna().all(axis=0)
        assets_today = assets_today[ok_hist_err.values]

    assets_today = pd.Index(assets_today).astype(str).sort_values()

    if len(assets_today) < min_assets:
        return None
    return assets_today


def compute_kappa_empirical(hist_err: pd.DataFrame, Lambda: np.ndarray, alpha: float = 0.95, ridge: float = 1e-8):
    """
    hist_err: (M, N) dataframe of past realized error vectors (rows=time, cols=assets) aligned to current assets
    Lambda:   (N, N) covariance matrix estimated from hist_err
    alpha:    desired coverage level (0.90, 0.95, 0.99)
    Returns:  kappa = sqrt(quantile_alpha(q)), where q = e^T Lambda^{-1} e
    """
    if hist_err is None or len(hist_err) < 5:
        return None

    E = hist_err.to_numpy(dtype=float)
    if not np.isfinite(E).all():
        return None

    N = Lambda.shape[0]
    L = Lambda + ridge * np.eye(N)

    try:
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        return None

    # q_i = e_i^T L_inv e_i
    q = np.einsum("ti,ij,tj->t", E, L_inv, E)

    q = q[np.isfinite(q)]
    if len(q) == 0:
        return None

    q_alpha = np.quantile(q, alpha)
    if not np.isfinite(q_alpha) or q_alpha <= 0:
        return None

    return float(np.sqrt(q_alpha))


def keep_topk_from_market_plus_others(hist_true: pd.DataFrame, assets: pd.Index, market: str, k: int = 10) -> pd.Index:
    tag = f"{market}:"
    mkt_assets = [a for a in assets if str(a).startswith(tag)]
    other_assets = [a for a in assets if not str(a).startswith(tag)]

    if len(mkt_assets) == 0:
        return assets  # no filtering possible / needed

    means = hist_true[mkt_assets].mean(axis=0).sort_values(ascending=False)
    keep_mkt = list(means.index[: min(k, len(means))])
    print("check")
    return pd.Index(other_assets + keep_mkt).astype(str).sort_values()


def run_backtest_from_tables(
    markets: list[str],
    model_name: str,
    year: int = 2025,
    prev_year=False,
    train_weeks: int = 52,
    error_weeks: int = 20,
    warmup_periods: int = 20,   # first K backtest periods (in 'year') use non-robust
    # kappa: float = 0.5,
    alpha: float = 0.9, ## quantile alpha for kappa selection (from data) - a=0 means no robust!
    # delta: float = 0.5,
    p: float=0.5,
    eps: float = 1e-8,
    out_dir: str | None = None,
    merged=False,
    top_dow30=False, ### choose top 10 dow30 before optimizaiton
):
    delta = (p/(1-p))**0.5
    markets_str = "-".join(markets)

    if merged:
        exp_path = os.path.join("data/prediction", f"{model_name}_{markets_str}_merged_expected_returns.csv")
        true_path = os.path.join("data/prediction", f"{model_name}_{markets_str}_merged_true_returns.csv")
        err_path = os.path.join("data/prediction", f"{model_name}_{markets_str}_merged_errors.csv")
    else:
        exp_path = os.path.join("data/prediction", f"{model_name}_{markets_str}_expected_returns.csv")
        true_path = os.path.join("data/prediction", f"{model_name}_{markets_str}_true_returns.csv")
        err_path = os.path.join("data/prediction", f"{model_name}_{markets_str}_errors.csv")     

    print("TRUE RETURNS from ", true_path)
    if not os.path.exists(exp_path):
        raise FileNotFoundError(exp_path)
    if not os.path.exists(true_path):
        raise FileNotFoundError(true_path)

    expected_df = _read_wide_csv(exp_path)
    true_df = _read_wide_csv(true_path)

    if os.path.exists(err_path):
        errors_df = _read_wide_csv(err_path)
    else:
        errors_df = true_df - expected_df

    # Ensure identical index ordering (union, then reindex)
    idx_all = expected_df.index.union(true_df.index).union(errors_df.index).sort_values()
    cols_all = expected_df.columns.union(true_df.columns).union(errors_df.columns)

    expected_df = expected_df.reindex(index=idx_all, columns=cols_all)
    true_df = true_df.reindex(index=idx_all, columns=cols_all)
    errors_df = errors_df.reindex(index=idx_all, columns=cols_all)

    # Backtest periods are only the requested year (but we can use pre-year rows for history)
    if isinstance(expected_df.index, pd.DatetimeIndex):
        if prev_year:
            backtest_idx = expected_df.index[expected_df.index.year.isin([year-1, year])] ## 2025 - 2024
        else:
            backtest_idx = expected_df.index[expected_df.index.year == year] ## 2025
        
    else:
        backtest_idx = expected_df.index

    if len(backtest_idx) == 0:
        raise ValueError(f"No backtest periods found for year={year}. Check CSV index parsing.")

    ports_rows = []
    port_summary_rows = []

    backtest_idx = list(backtest_idx)

    for j, t in enumerate(backtest_idx):
        # history up to t-1
        hist_true = true_df.loc[:t].iloc[:-1].tail(train_weeks)
        hist_err = errors_df.loc[:t].iloc[:-1].tail(error_weeks)

        exp_row = expected_df.loc[t]
        true_row = true_df.loc[t]

        assets = _align_universe_for_period(
            expected_row=exp_row,
            true_row=true_row,
            hist_true=hist_true if len(hist_true) else None,
            hist_err=hist_err if len(hist_err) else None,
            min_assets=2,
        )
        if assets is None:
            continue

        if top_dow30:
            assets = keep_topk_from_market_plus_others(hist_true, assets, market="dow30", k=10)
            if len(assets) < 2:
                continue
        
        mu_hat = exp_row.loc[assets].to_numpy(dtype=float)
        r_real = true_row.loc[assets].to_numpy(dtype=float)

        Sigma = _safe_cov(hist_true[assets].to_numpy(dtype=float), eps=eps)
        if Sigma is None:
            continue

        Lambda = _safe_cov(hist_err[assets].to_numpy(dtype=float), eps=eps)
        
        ## Kappa selection!
        kappa = 0.0
        if (j >= warmup_periods) and (Lambda is not None):
            kappa = compute_kappa_empirical(hist_err[assets], Lambda, alpha=alpha)  # <-- new
            if kappa is None or alpha==0:
                kappa = 0.0  # fallback to fixed
                print("ERROR on KAPPA!")

        if (j >= warmup_periods) and (Lambda is not None):
            w_opt, obj = solve_markowitz_robust(
                assets=list(assets),
                expected_returns=mu_hat,
                Sigma=Sigma,
                Lambda=Lambda,
                kappa=kappa,
                delta=delta,
                verbose=True
            )
            is_robust = True
        else:
            Lambda0 = 1e-12 * np.eye(len(assets))
            w_opt, obj = solve_markowitz_robust(
                assets=list(assets),
                expected_returns=mu_hat,
                Sigma=Sigma,
                Lambda=Lambda0,
                kappa=0.0,
                delta=delta,
                verbose=True
            )
            is_robust = False

        expected_port_ret = float(np.dot(w_opt, mu_hat))
        realized_port_ret = float(np.dot(w_opt, r_real))

        min_invest_w = 1e-6  # or 1e-5 to match your zeroing threshold
        n_invested = int(np.sum(np.abs(w_opt) > min_invest_w))

        port_summary_rows.append({
            "period": t,
            "expected_portfolio_return": expected_port_ret,
            "realized_portfolio_return": realized_port_ret,
            "objective": float(obj),
            "is_robust": int(is_robust),
            "kappa": float(kappa),
            "delta": float(delta),
            "n_assets": int(len(assets)),
            "n_invested": n_invested,
            "sum_w": float(np.sum(w_opt)),
        })

        for i, a in enumerate(assets):
            ports_rows.append({
                "period": t,
                "asset": a,
                "weight": float(w_opt[i]),
                "expected_portfolio_return": expected_port_ret,
                "realized_portfolio_return": realized_port_ret,
                "objective": float(obj),
                "is_robust": int(is_robust),
            })

    ports_df = pd.DataFrame(ports_rows)
    port_summary_df = pd.DataFrame(port_summary_rows)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if merged:
            out_ports = os.path.join(out_dir, f"{model_name}_permarket_{markets_str}_ports_{year}.csv")
            out_summary = os.path.join(out_dir, f"{model_name}_permarket_{markets_str}_port_summary_{year}.csv")
            out_summary_e = os.path.join(out_dir, f"{model_name}_permarket_{markets_str}_port_summary_{year}.xlsx")
            
        else:
            out_ports = os.path.join(out_dir, f"a{alpha}_p{p}_w{warmup_periods}_n{len(assets)}_{model_name}_singlecrossmarket_details_{year}.csv")
            out_summary = os.path.join(out_dir, f"a{alpha}_p{p}_w{warmup_periods}_n{len(assets)}_{model_name}_singlecrossmarket_summary_{year}.csv")
            out_summary_e = os.path.join(out_dir, f"a{alpha}_p{p}_w{warmup_periods}_n{len(assets)}_{model_name}_singlecrossmarket_summary_{year}.xlsx")
            # out_ports = os.path.join(out_dir, f"{model_name}_singlecrossmarket_{markets_str}_ports_{year}.csv")
            # out_summary = os.path.join(out_dir, f"{model_name}_singlecrossmarket_{markets_str}_port_summary_{year}.csv")
            # out_summary_e = os.path.join(out_dir, f"{model_name}_singlecrossmarket_{markets_str}_port_summary_{year}.xlsx")
        ports_df.to_csv(out_ports, index=False, float_format="%.8f")
        port_summary_df.to_csv(out_summary, index=False, float_format="%.8f")
        port_summary_df.to_excel(out_summary_e, index=False, float_format="%.8f")
        print("Saved:")
        print(" ", out_ports, ports_df.shape)
        print(" ", out_summary, port_summary_df.shape)

    return ports_df, port_summary_df

## 2025
# run_backtest_from_tables(
#     # markets=['dow30'],
#     markets=['dow30', 'commodities', 'bonds'],
#     model_name='MLP',
#     year=2025,
#     train_weeks=60,
#     error_weeks=60,
#     warmup_periods=60,
#     # kappa=0.5,
#     alpha=0.5,
#     # delta=0.5,
#     p=0.5,
#     eps=1e-8,
#     out_dir='outputs/backtest',
#     merged=False,
#     top_dow30=True,
# )

alpha_set = 
p_set = ()
w_set = (20,40,60,80)
top10_set = 
# ## 2024-25
run_backtest_from_tables(
    # markets=['dow30'],
    markets=['dow30', 'commodities', 'bonds'],
    model_name='MLP',
    year=2025,
    prev_year=True,
    train_weeks=60,
    error_weeks=60,
    warmup_periods=60,
    # kappa=0.5,
    alpha=0.5,
    # delta=0.5,
    p=0.5,
    eps=1e-8,
    out_dir='outputs/backtest',
    merged=False,
    top_dow30=True
)

