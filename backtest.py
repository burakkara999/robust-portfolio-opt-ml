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
    train_periods: int = 52,      # history length for Sigma (true returns)
    warmup_period: int = 20,      # history length for Lambda/kappa (residuals), BEFORE start_date
    alpha: float = 0.9,           # quantile for kappa; alpha=0 => no robust
    p: float = 0.5,               # delta = sqrt(p/(1-p))
    eps: float = 1e-8,
    start_date: str | None = None,   # trading starts here (money initialized here)
    end_date: str | None = None,
    initial_money: float = 1000.0,
    out_dir: str | None = None,
):
    """
    Trading and robust start at `start_date`.
    Warmup is PRE-start_date history: we use the last `warmup_period` residual rows before start_date
    to estimate Lambda and compute kappa on the first trading date.
    """

    markets_str = "-".join(markets)

    pred_dir = os.path.join("data", "prediction")
    true_dir = os.path.join("data", "log_returns")

    mu_path   = os.path.join(pred_dir, f"{model_name}_{markets_str}_log_returns.csv")
    err_path  = os.path.join(pred_dir, f"{model_name}_{markets_str}_residuals.csv")
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

    # ---- Align dates/columns on full intersection
    idx_all = mu_df.index.intersection(err_df.index).intersection(true_df.index).sort_values()
    if len(idx_all) == 0:
        raise ValueError("No overlapping dates across prediction/residual/true tables.")

    cols = mu_df.columns.intersection(err_df.columns).intersection(true_df.columns).astype(str)
    if len(cols) == 0:
        raise ValueError("No overlapping assets (columns) across the three tables.")

    mu_df = mu_df.loc[idx_all, cols]
    err_df = err_df.loc[idx_all, cols]
    true_df = true_df.loc[idx_all, cols]

    # ---- Trading start/end
    trade_start_req = pd.to_datetime(start_date) if start_date is not None else idx_all[0]
    trade_end = pd.to_datetime(end_date) if end_date is not None else idx_all[-1]

    # snap to first available date >= requested
    pos0 = idx_all.get_indexer([trade_start_req], method="bfill")[0]
    if pos0 < 0:
        raise ValueError(f"start_date={trade_start_req} is after the last available date in tables.")
    trade_start = idx_all[pos0]

    # ---- We need PRE-start history for BOTH Sigma and Lambda/kappa
    buffer_len = max(train_periods, warmup_period)
    pos_buffer = max(0, pos0 - buffer_len)
    buffer_start = idx_all[pos_buffer]

    # iterate from buffer_start (for history), but RECORD only [trade_start, trade_end]
    idx_loop = idx_all[(idx_all >= buffer_start) & (idx_all <= trade_end)]
    if len(idx_loop) == 0:
        raise ValueError("No dates in idx_loop after applying buffer_start/trade_end.")

    delta = math.sqrt(p / (1.0 - p))

    details_rows = []
    summary_rows = []

    # money is initialized AT trade_start (not after warmup)
    money = float(initial_money)
    expected_money = float(initial_money)

    for t in idx_loop:
        is_trading_date = (t >= trade_start)

        # history up to t-1
        hist_true = true_df.loc[:t].iloc[:-1].tail(train_periods)
        hist_err = err_df.loc[:t].iloc[:-1].tail(warmup_period)

        mu_row = mu_df.loc[t]
        r_row = true_df.loc[t]

        assets = pick_assets_for_t(
            mu_row=mu_row,
            r_row=r_row,
            hist_true=hist_true,
            hist_err=hist_err,
            min_assets=2,
        )
        if assets is None:
            continue

        mu_hat = mu_row.loc[assets].to_numpy(dtype=float)
        r_real = r_row.loc[assets].to_numpy(dtype=float)

        Sigma = safe_cov(hist_true[assets].to_numpy(dtype=float), eps=eps)
        if Sigma is None:
            # not enough return history
            continue

        # Robust starts immediately at trade_start, BUT only if we actually have warmup_period residual history
        have_err_hist = (len(hist_err) >= warmup_period)
        is_robust = bool(is_trading_date and (alpha > 0) and have_err_hist)

        Lambda = None
        kappa = 0.0

        if is_robust:
            Lambda = safe_cov(hist_err[assets].to_numpy(dtype=float), eps=eps)
            if Lambda is None:
                is_robust = False

        if is_robust:
            kappa_val = compute_kappa_empirical(hist_err[assets], Lambda, alpha=alpha)
            if (kappa_val is None) or (not np.isfinite(kappa_val)):
                is_robust = False
            else:
                kappa = float(kappa_val)

        # Solve
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
            # non-robust fallback
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
        n_invested = int(np.sum(np.abs(w_opt) > 1e-6))

        # Update money ONLY during trading horizon (>= trade_start)
        if is_trading_date:
            money *= float(np.exp(real_port_ret))
            expected_money *= float(np.exp(exp_port_ret))

            # trade_j: 0 at trade_start, 1 next trading date, ...
            trade_j = int(np.searchsorted(idx_all, t) - np.searchsorted(idx_all, trade_start))

            summary_rows.append({
                "period": t,
                "trade_j": trade_j,
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
                "have_err_hist": int(have_err_hist),
            })

            for a, w in zip(assets, w_opt):
                details_rows.append({
                    "period": t,
                    "trade_j": trade_j,
                    "asset": a,
                    "weight": float(w),
                    "expected_portfolio_return": exp_port_ret,
                    "realized_portfolio_return": real_port_ret,
                    "objective": float(obj),
                    "is_robust": int(is_robust),
                })

    details_df = pd.DataFrame(details_rows)
    summary_df = pd.DataFrame(summary_rows)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tag = (
            f"{model_name}_{markets_str}_a{alpha}_p{p}"
            f"_train{train_periods}_warm{warmup_period}"
            # f"_{trade_start.date()}_{trade_end.date()}"
        )

        out_details = os.path.join(out_dir, f"{tag}_details.csv")
        out_summary = os.path.join(out_dir, f"{tag}_summary.csv")
        out_summary_xlsx = os.path.join(out_dir, f"{tag}_summary.xlsx")

        details_df.to_csv(out_details, index=False, float_format="%.10f")
        summary_df.to_csv(out_summary, index=False, float_format="%.10f")
        summary_df.to_excel(out_summary_xlsx, index=False)

        print("Saved:")
        print(" ", out_details, details_df.shape)
        print(" ", out_summary, summary_df.shape)

    # sanity check: first trading row should be trade_start
    if len(summary_df) > 0:
        print("First recorded trading date:", summary_df["period"].iloc[0])

    return details_df, summary_df


# if __name__ == "__main__":
#     # Example (matches your screenshot naming style):
#     #   data/prediction/SM60_crypto-commodities-bonds_log_returns.csv
#     #   data/prediction/SM60_crypto-commodities-bonds_residuals.csv
#     #   data/log_returns/log_returns_crypto-commodities-bonds.csv
#     details_df, summary_df = run_backtest(
#         markets=["crypto", "commodities", "bonds"],
#         model_name="SM10",
#         train_periods=20,
#         warmup_period=20,
#         alpha=0.3,
#         p=0.3,
#         start_date="2025-01-03",
#         end_date=None,
#         initial_money=1000.0,
#         out_dir="outputs/backtests",
#     )
#     print(summary_df.tail(3))

## GRID search below
def _max_drawdown_from_money(money_series: pd.Series) -> float:
    """money_series: positive wealth series indexed by time (can contain NaN at start)."""
    x = money_series.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    peak = np.maximum.accumulate(x)
    dd = (x / peak) - 1.0
    return float(dd.min())  # negative number

def _summarize_run(summary_df: pd.DataFrame) -> dict:
    """
    summary_df must contain:
      - period
      - realized_portfolio_return
      - expected_portfolio_return
      - money
      - expected_money
      - is_robust
    """
    if summary_df is None or len(summary_df) == 0:
        return {
            "n_periods": 0,
            "last_money": np.nan,
            "last_expected_money": np.nan,
            "cum_realized_logret": np.nan,
            "cum_expected_logret": np.nan,
            "mean_realized_logret": np.nan,
            "std_realized_logret": np.nan,
            "sharpe_like": np.nan,
            "max_drawdown": np.nan,
            "robust_share": np.nan,
        }

    sdf = summary_df.sort_values("period").copy()

    # returns are log-returns in your code (you use exp(ret) for wealth)
    r = pd.to_numeric(sdf["realized_portfolio_return"], errors="coerce")
    e = pd.to_numeric(sdf["expected_portfolio_return"], errors="coerce")

    last_money = pd.to_numeric(sdf["money"], errors="coerce").dropna()
    last_expected = pd.to_numeric(sdf["expected_money"], errors="coerce").dropna()

    cum_r = float(r.sum(skipna=True))
    cum_e = float(e.sum(skipna=True))

    mean_r = float(r.mean(skipna=True))
    std_r = float(r.std(skipna=True, ddof=1))
    sharpe_like = float(mean_r / std_r) if (np.isfinite(std_r) and std_r > 0) else np.nan

    max_dd = _max_drawdown_from_money(pd.to_numeric(sdf["money"], errors="coerce"))

    robust_share = float(pd.to_numeric(sdf.get("is_robust", 0), errors="coerce").mean()) \
        if "is_robust" in sdf.columns else np.nan

    return {
        "n_periods": int(len(sdf)),
        "first_period": sdf["period"].iloc[0],
        "last_period": sdf["period"].iloc[-1],
        "last_money": float(last_money.iloc[-1]) if len(last_money) else np.nan,
        "last_expected_money": float(last_expected.iloc[-1]) if len(last_expected) else np.nan,
        "cum_realized_logret": cum_r,
        "cum_expected_logret": cum_e,
        "mean_realized_logret": mean_r,
        "std_realized_logret": std_r,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_dd,
        "robust_share": robust_share,
    }

# ----------------------------
# Grid search + results table
# ----------------------------
a_set = [0.0, 0.1, 0.3]
p_set = [0.1, 0.3, 0.5]
models = ["SM10", "SM30", "SM60"]

results = []

for model_name in models:
    for a in a_set:
        for p in p_set:
            print(f"Running backtest for model={model_name}, alpha={a}, p={p}")
            details_df, summary_df = run_backtest(
                markets=["crypto", "commodities", "bonds"],
                model_name=model_name,
                train_periods=20,
                warmup_period=20,
                alpha=a,
                p=p,
                start_date="2025-01-03",
                end_date=None,
                initial_money=1000.0,
                out_dir="outputs/backtests",
            )

            row = {
                "model": model_name,
                "alpha": float(a),
                "p": float(p),
            }
            row.update(_summarize_run(summary_df))
            results.append(row)

grid_df = pd.DataFrame(results)

# Sort how you like (e.g., best realized money first)
grid_df = grid_df.sort_values(["last_money", "last_expected_money"], ascending=False)

# Save one Excel file
os.makedirs("outputs/backtests", exist_ok=True)
out_xlsx = os.path.join("outputs/backtests", "grid_search_results.xlsx")
grid_df.to_excel(out_xlsx, index=False)

print("Saved grid table to:", out_xlsx)
print(grid_df.head(10))