#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd


# -----------------------------
# IO
# -----------------------------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df.columns = df.columns.astype(str)
    return df.sort_index()


def align_pred_true(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pred_df.index.union(true_df.index).sort_values()
    cols = pred_df.columns.union(true_df.columns)
    pred_df = pred_df.reindex(index=idx, columns=cols)
    true_df = true_df.reindex(index=idx, columns=cols)
    return pred_df, true_df


# -----------------------------
# Metrics
# -----------------------------
def _safe_div(a, b, eps=1e-12):
    return a / (np.abs(b) + eps)

def _rankdata(x: np.ndarray) -> np.ndarray:
    # simple rank (ties get average rank via pandas)
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # assume already filtered finite
    err = y_true - y_pred

    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    mean_abs_true = float(np.mean(np.abs(y_true)))
    mean_abs_pred = float(np.mean(np.abs(y_pred)))

    # normalized errors (scale-free)
    nmae = float(mae / (mean_abs_true + 1e-12))
    nrmse = float(rmse / (mean_abs_true + 1e-12))

    # R2 (guard)
    ybar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - ybar) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # MAPE + sMAPE (guard zeros)
    mape = float(np.mean(_safe_div(np.abs(err), y_true)))
    smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-12)))

    # Pearson correlation (guard)
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    # directional accuracy
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # Rank IC (Spearman)
    if len(y_true) >= 2:
        rt = _rankdata(y_true)
        rp = _rankdata(y_pred)
        if np.std(rt) > 0 and np.std(rp) > 0:
            rank_ic = float(np.corrcoef(rt, rp)[0, 1])
        else:
            rank_ic = np.nan
    else:
        rank_ic = np.nan

    return {
        "n": int(len(y_true)),
        "mean_abs_true": mean_abs_true,
        "mean_abs_pred": mean_abs_pred,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nmae": nmae,
        "nrmse": nrmse,
        "mape": mape,
        "smape": smape,
        "corr": corr,
        "r2": r2,
        "dir_acc": dir_acc,
        "rank_ic": rank_ic,
    }


def pooled_metrics(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> dict:
    P = pred_df.to_numpy(dtype=float)
    T = true_df.to_numpy(dtype=float)
    mask = np.isfinite(P) & np.isfinite(T)
    y_pred = P[mask]
    y_true = T[mask]
    if len(y_true) == 0:
        return {"n": 0}
    return compute_metrics(y_true, y_pred)


def per_period_metrics(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in pred_df.index:
        p = pred_df.loc[t].to_numpy(dtype=float)
        y = true_df.loc[t].to_numpy(dtype=float)
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() == 0:
            continue
        met = compute_metrics(y[m], p[m])
        rows.append({"period": t, **met})
    return pd.DataFrame(rows).sort_values("period")


def per_asset_metrics(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in pred_df.columns:
        p = pred_df[a].to_numpy(dtype=float)
        y = true_df[a].to_numpy(dtype=float)
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() == 0:
            continue
        met = compute_metrics(y[m], p[m])
        rows.append({"asset": a, **met})
    return pd.DataFrame(rows).sort_values("asset")


def evaluate_one(model_name: str, markets_str: str, pred_dir: str, out_dir: str) -> dict:
    exp_path = os.path.join(pred_dir, f"{model_name}_{markets_str}_expected_returns.csv")
    true_path = os.path.join(pred_dir, f"{model_name}_{markets_str}_true_returns.csv")
    # exp_path = os.path.join(pred_dir, f"{model_name}_{markets_str}_merged_expected_returns.csv")
    # true_path = os.path.join(pred_dir, f"{model_name}_{markets_str}_merged_true_returns.csv")

    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Missing expected_returns: {exp_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing true_returns: {true_path}")

    pred_df = read_wide_csv(exp_path)
    true_df = read_wide_csv(true_path)
    pred_df, true_df = align_pred_true(pred_df, true_df)

    pooled = pooled_metrics(pred_df, true_df)
    by_period = per_period_metrics(pred_df, true_df)
    by_asset = per_asset_metrics(pred_df, true_df)

    os.makedirs(out_dir, exist_ok=True)
    pooled_out = os.path.join(out_dir, f"{model_name}_{markets_str}_pooled_metrics.xlsx")
    per_out = os.path.join(out_dir, f"{model_name}_{markets_str}_period_metrics.xlsx")
    asset_out = os.path.join(out_dir, f"{model_name}_{markets_str}_asset_metrics.xlsx")

    pd.DataFrame([{"model": model_name, "markets": markets_str, **pooled}]).to_excel(pooled_out, index=False)
    by_period.to_excel(per_out, index=False)
    by_asset.to_excel(asset_out, index=False)

    print("Saved:")
    print(" ", pooled_out)
    print(" ", per_out)
    print(" ", asset_out)

    return {"model": model_name, "markets": markets_str, **pooled}


def main():
    pred_dir = "data/prediction"
    out_dir = "outputs/metrics"
    markets_str = "dow30-commodities-bonds"

    summaries = []
    for m in ["MLP", "SM60"]:
    # for m in ["MLP"]:
        summaries.append(evaluate_one(m, markets_str, pred_dir, out_dir))

    # add skill vs baseline (SM60) into summary
    summary_df = pd.DataFrame(summaries).set_index("model")

    if "SM60" in summary_df.index:
        base_mae = float(summary_df.loc["SM60", "mae"])
        base_rmse = float(summary_df.loc["SM60", "rmse"])
        for model in summary_df.index:
            summary_df.loc[model, "skill_mae_vs_SM60"] = 1.0 - float(summary_df.loc[model, "mae"]) / (base_mae + 1e-12)
            summary_df.loc[model, "skill_rmse_vs_SM60"] = 1.0 - float(summary_df.loc[model, "rmse"]) / (base_rmse + 1e-12)

    summary_df = summary_df.reset_index()
    summary_path = os.path.join(out_dir, f"summary_{markets_str}.xlsx")
    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_excel(summary_path, index=False)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()