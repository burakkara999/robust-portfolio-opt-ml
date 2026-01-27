"""
Backtest robust Markowitz using SAVED prediction tables.

Reads:
  - data/prediction/{model_name}_{markets}_log_returns.csv      (expected returns, wide; index=date, cols=assets)
  - data/prediction/{model_name}_{markets}_residuals.csv        (errors = true - predicted, wide; index=date, cols=assets)
  - data/log_returns/log_returns_{markets}.csv                  (true returns, wide; index=date, cols=assets)

Notes:
- Sigma (return covariance) is estimated from the last `train_periods` rows of true returns before t.
- Lambda (error covariance) is estimated from the last `warmup_period` rows of residuals before t.
- For the first `warmup_period` backtest points, we run NON-robust (kappa=0, tiny Lambda).
"""

import os
import math
import numpy as np
import pandas as pd

from portfolio_models import solve_markowitz_robust


# -----------------------------
# IO helpers
# -----------------------------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # parse dates if possible
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df.columns = df.columns.astype(str)
    return df.sort_index()


def safe_cov(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    """
    mat: (T, N), rows=time, cols=assets.
    Returns covariance (N, N) with ridge; None if insufficient samples.
    """
    if mat.ndim != 2:
        raise ValueError("mat must be 2D")
    if mat.shape[0] < 2:
        return None
    C = np.cov(mat, rowvar=False)
    return C + eps * np.eye(C.shape[0])


# -----------------------------
# Robust radius (empirical kappa)
# -----------------------------
def compute_kappa_empirical(
    hist_err: pd.DataFrame,
    Lambda: np.ndarray,
    alpha: float = 0.95,
    ridge: float = 1e-8,
) -> float | None:
    """
    hist_err: (M,N) past error vectors (rows=time).
    Lambda:   (N,N) error covariance estimate.
    Returns kappa = sqrt( quantile_alpha( e^T Lambda^{-1} e ) ).
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

    q = np.einsum("ti,ij,tj->t", E, L_inv, E)
    q = q[np.isfinite(q)]
    if len(q) == 0:
        return None

    q_alpha = np.quantile(q, alpha)
    if not np.isfinite(q_alpha) or q_alpha <= 0:
        return None

    return float(np.sqrt(q_alpha))


# -----------------------------
# Universe alignment
# -----------------------------
def pick_assets_for_t(
    mu_row: pd.Series,
    r_row: pd.Series,
    hist_true: pd.DataFrame,
    hist_err: pd.DataFrame,
    min_assets: int = 2,
) -> pd.Index | None:
    """
    Keep assets that have:
      - finite mu and r at time t
      - no NA in history windows used for Sigma/Lambda
    """
    assets = mu_row.index.astype(str)

    r_aligned = r_row.reindex(assets)
    ok_today = np.isfinite(mu_row.values) & np.isfinite(r_aligned.values)
    assets = pd.Index(assets[ok_today]).astype(str)

    if len(assets) == 0:
        return None

    if hist_true is not None and len(hist_true) > 0:
        ok_hist_true = hist_true[assets].notna().all(axis=0)
        assets = assets[ok_hist_true.values]

    if hist_err is not None and len(hist_err) > 0:
        ok_hist_err = hist_err[assets].notna().all(axis=0)
        assets = assets[ok_hist_err.values]

    assets = pd.Index(assets).astype(str).sort_values()
    if len(assets) < min_assets:
        return None
    return assets


# -----------------------------
# Main backtest
# -----------------------------
def run_backtest(
    markets: list[str],
    model_name: str,
    train_periods: int = 52,    # history length for Sigma(covariance)
    warmup_period: int = 20,   # also used as residual-history length
    alpha: float = 0.9,        # quantile for kappa; alpha=0 => no robust
    p: float = 0.5,            # robust Markowitz parameter -> delta = sqrt(p/(1-p))
    eps: float = 1e-8,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_money: float = 1000.0,
    out_dir: str | None = None,
):
    markets_str = "-".join(markets)

    pred_dir = os.path.join("data", "prediction")
    true_dir = os.path.join("data", "log_returns")

    mu_path = os.path.join(pred_dir, f"{model_name}_{markets_str}_log_returns.csv")
    err_path = os.path.join(pred_dir, f"{model_name}_{markets_str}_residuals.csv")
    true_path = os.path.join(true_dir, f"log_returns_{markets_str}.csv")

    if not os.path.exists(mu_path):
        raise FileNotFoundError(mu_path)
    if not os.path.exists(err_path):
        raise FileNotFoundError(err_path)
    if not os.path.exists(true_path):
        raise FileNotFoundError(true_path)

    mu_df = read_wide_csv(mu_path)       # predicted/expected returns
    err_df = read_wide_csv(err_path)     # residuals = true - predicted
    true_df = read_wide_csv(true_path)   # realized returns

    # Align on common dates + common assets (keep wide structure)
    idx = mu_df.index.intersection(err_df.index).intersection(true_df.index)
    if start_date is not None:
        idx = idx[idx >= pd.to_datetime(start_date)]
    if end_date is not None:
        idx = idx[idx <= pd.to_datetime(end_date)]
    idx = idx.sort_values()

    if len(idx) == 0:
        raise ValueError("No overlapping dates after alignment/filtering.")

    cols = mu_df.columns.intersection(err_df.columns).intersection(true_df.columns).astype(str)
    if len(cols) == 0:
        raise ValueError("No overlapping assets (columns) across the three tables.")

    mu_df = mu_df.loc[idx, cols]
    err_df = err_df.loc[idx, cols]
    true_df = true_df.loc[idx, cols]

    delta = math.sqrt(p / (1.0 - p))

    details = []
    summary = []

    money = float(initial_money)
    expected_money = float(initial_money)

    for j, t in enumerate(idx):
        # history up to t-1
        hist_true = true_df.loc[:t].iloc[:-1].tail(train_periods)
        hist_err = err_df.loc[:t].iloc[:-1].tail(warmup_period)

        mu_row = mu_df.loc[t]
        r_row = true_df.loc[t]

        assets = pick_assets_for_t(mu_row, r_row, hist_true, hist_err, min_assets=2)
        if assets is None:
            continue

        mu_hat = mu_row.loc[assets].to_numpy(dtype=float)
        r_real = r_row.loc[assets].to_numpy(dtype=float)

        Sigma = safe_cov(hist_true[assets].to_numpy(dtype=float), eps=eps)
        if Sigma is None:
            continue

        # robust vs non-robust decision
        is_robust = (j >= warmup_period) and (alpha > 0)

        if is_robust:
            Lambda = safe_cov(hist_err[assets].to_numpy(dtype=float), eps=eps)
            if Lambda is None:
                is_robust = False

        if is_robust:
            kappa = compute_kappa_empirical(hist_err[assets], Lambda, alpha=alpha)
            if (kappa is None) or (not np.isfinite(kappa)):
                # fallback: disable robust if kappa cannot be computed
                is_robust = False

        if is_robust:
            w_opt, obj = solve_markowitz_robust(
                assets=list(assets),
                expected_returns=mu_hat,
                Sigma=Sigma,
                Lambda=Lambda,
                kappa=float(kappa),
                delta=float(delta),
                verbose=False,
            )
        else:
            # non-robust: kappa=0 and near-zero Lambda
            Lambda0 = 1e-12 * np.eye(len(assets))
            w_opt, obj = solve_markowitz_robust(
                assets=list(assets),
                expected_returns=mu_hat,
                Sigma=Sigma,
                Lambda=Lambda0,
                kappa=0.0,
                delta=float(delta),
                verbose=False,
            )
            kappa = 0.0

        exp_port_ret = float(np.dot(w_opt, mu_hat))
        real_port_ret = float(np.dot(w_opt, r_real))

        money *= float(np.exp(real_port_ret))
        expected_money *= float(np.exp(exp_port_ret))

        n_invested = int(np.sum(np.abs(w_opt) > 1e-6))

        summary.append({
            "period": t,
            "expected_portfolio_return": exp_port_ret,
            "realized_portfolio_return": real_port_ret,
            "objective": float(obj),
            "is_robust": int(is_robust),
            "kappa": float(kappa),
            "delta": float(delta),
            "n_assets": int(len(assets)),
            "n_invested": n_invested,
            "sum_w": float(np.sum(w_opt)),
            "money": float(money),
            "expected_money": float(expected_money),
        })

        for a, w in zip(assets, w_opt):
            details.append({
                "period": t,
                "asset": a,
                "weight": float(w),
                "expected_portfolio_return": exp_port_ret,
                "realized_portfolio_return": real_port_ret,
                "objective": float(obj),
                "is_robust": int(is_robust),
            })

    details_df = pd.DataFrame(details)
    summary_df = pd.DataFrame(summary)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{model_name}_{markets_str}_a{alpha}_p{p}_train{train_periods}_warm{warmup_period}"

        out_details = os.path.join(out_dir, f"{tag}_details.csv")
        out_summary = os.path.join(out_dir, f"{tag}_summary.csv")
        out_summary_xlsx = os.path.join(out_dir, f"{tag}_summary.xlsx")

        details_df.to_csv(out_details, index=False, float_format="%.10f")
        summary_df.to_csv(out_summary, index=False, float_format="%.10f")
        summary_df.to_excel(out_summary_xlsx, index=False)

        print("Saved:")
        print(" ", out_details, details_df.shape)
        print(" ", out_summary, summary_df.shape)

    return details_df, summary_df


if __name__ == "__main__":
    # Example (matches your screenshot naming style):
    #   data/prediction/SM60_crypto-commodities-bonds_log_returns.csv
    #   data/prediction/SM60_crypto-commodities-bonds_residuals.csv
    #   data/log_returns/log_returns_crypto-commodities-bonds.csv
    details_df, summary_df = run_backtest(
        markets=["crypto", "commodities", "bonds"],
        model_name="SM60",
        train_periods=20,
        warmup_period=20,
        alpha=0.0,
        p=0.5,
        start_date="2025-01-03",
        end_date=None,
        initial_money=1000.0,
        out_dir="outputs/backtest_SM60_simple",
    )
    print(summary_df.tail(3))