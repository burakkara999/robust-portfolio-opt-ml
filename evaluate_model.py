# evaluate_model.py

import numpy as np
import pandas as pd

from asset_data_module import read_close_prices_all_merged
from features import make_feature_windows


## MODEL 1 - MLP
from models.mlp_returns import fit_predict_mlp

## MODEL 2 - per-asset MLP
from models.mlp_returns_per_asset import fit_predict_mlp_per_asset

## benchmark model -- rolling (sample) mean
from models.rolling_mean import fit_predict_rolling_mean


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs(err) / denom))
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)))

    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # Correlation / R2 (guard small N / zero variance)
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    denom_r2 = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom_r2) if denom_r2 > 0 else np.nan

    return {
        "N": int(len(y_true)),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape,
        "DirAcc": dir_acc,
        "Corr": corr,
        "R2": r2,
    }


def evaluate_walk_forward(
    markets: list[str],
    model_fit_predict_fn,  # (train_windows, test_window) -> pd.Series(index=assets, values=y_pred)
    lookback_weeks=5,
    horizon_weeks=1,
    train_weeks=52,
    start_date="2022-01-01",
    end_date="2025-11-30",
    out_preds_csv=None,
    out_week_metrics_csv=None,
    out_asset_metrics_csv=None,
):
    # ---- Load + windows (same as backtest_runner) ----
    _, close_df = read_close_prices_all_merged(markets, after_date=start_date)
    close_df = close_df.loc[:end_date]

    rolling = make_feature_windows(
        close_prices=close_df,
        lookback=lookback_weeks,
        horizon=horizon_weeks,
    )

    preds_rows = []
    week_rows = []

    for k in range(train_weeks, len(rolling)):
        train_windows = rolling[k - train_weeks : k]
        test_window = rolling[k]

        # same safe alignment as backtest_runner
        common_assets = test_window["y_ret"].index
        for w in train_windows:
            common_assets = common_assets.intersection(w["y_ret"].index)
        common_assets = common_assets.sort_values()

        if len(common_assets) < 2:
            continue

        y_pred = model_fit_predict_fn(train_windows, test_window)  # Series
        y_true = test_window["y_ret"]

        y_pred = y_pred.loc[common_assets].astype(float)
        y_true = y_true.loc[common_assets].astype(float)

        err = (y_true - y_pred)

        # store per-asset predictions
        for a in common_assets:
            preds_rows.append(
                {
                    "t0": test_window["t0"],
                    "t1": test_window["t1"],
                    "asset": a,
                    "y_true": float(y_true.loc[a]),
                    "y_pred": float(y_pred.loc[a]),
                    "err": float(err.loc[a]),
                }
            )

        # cross-sectional metrics for this week
        m = _metrics(y_true.to_numpy(), y_pred.to_numpy())
        m.update({"t0": test_window["t0"], "t1": test_window["t1"], "n_assets": int(len(common_assets))})
        week_rows.append(m)

    preds_df = pd.DataFrame(preds_rows)
    week_metrics_df = pd.DataFrame(week_rows)

    # overall metrics across all (week,asset) rows
    overall = _metrics(preds_df["y_true"].to_numpy(), preds_df["y_pred"].to_numpy())
    overall_df = pd.DataFrame([{"scope": "overall", **overall}])

    # per-asset metrics across time
    asset_rows = []
    for a, g in preds_df.groupby("asset"):
        mm = _metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        asset_rows.append({"asset": a, **mm})
    asset_metrics_df = pd.DataFrame(asset_rows).sort_values("RMSE", ascending=True)

    # save
    if out_preds_csv:
        preds_df.to_csv(out_preds_csv, index=False)
    if out_week_metrics_csv:
        pd.concat([overall_df, week_metrics_df], ignore_index=True).to_csv(out_week_metrics_csv, index=False)
    if out_asset_metrics_csv:
        asset_metrics_df.to_csv(out_asset_metrics_csv, index=False)

    return preds_df, week_metrics_df, asset_metrics_df, overall_df


if __name__ == "__main__":
    # ---- Choose your markets + model here (like main.py) ----
    # markets_chosen = ["commodities", "bonds"]
    markets_chosen = ["commodities"]
    out_name = "rolling_mean_comm"
    model = fit_predict_rolling_mean

    preds_df, week_df, asset_df, overall_df = evaluate_walk_forward(
        markets=markets_chosen,
        model_fit_predict_fn=model,  # swap to mlp_per_asset_fit_predict if you want
        train_weeks=52,
        start_date="2022-01-01",
        end_date="2025-11-30",
        out_preds_csv=f"outputs/model_eval/{out_name}_preds.csv",
        out_week_metrics_csv=f"outputs/model_eval/{out_name}_week_metrics.csv",
        out_asset_metrics_csv=f"outputs/model_eval/{out_name}_asset_metrics.csv",
    )

    print("OVERALL:")
    print(overall_df.to_string(index=False))
    print("\nLAST 5 WEEKS (cross-sectional):")
    print(week_df.tail(5).to_string(index=False))
    print("\nTOP 10 ASSETS by RMSE:")
    print(asset_df.head(10).to_string(index=False))