from backtest_runner import run_walk_forward
from dataset_builders import X_from_past_returns, y_from_next_week_return
from models.mlp_returns import fit_predict as mlp_returns_fit_predict

preds_df, ports_df = run_walk_forward(
    markets=["dow30"],
    model_fit_predict_fn=mlp_returns_fit_predict,
    X_fn=lambda w: X_from_past_returns(w, lookback_days=25),
    y_fn=y_from_next_week_return,
    train_weeks=52,
    error_weeks=20,
    start_date="2022-01-01",
    end_date="2025-11-30",
    kappa=1,
    delta=3,
    out_preds_csv="outputs/preds_dow30_mlp_returns.csv",
    out_ports_csv="outputs/ports_dow30_mlp_returns.csv",
    out_ports_summary_csv="outputs/ports_summary_dow30_mlp_returns.csv",
)
print(preds_df.head())
print(ports_df.head())