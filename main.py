from backtest_runner import run_walk_forward
from dataset_builders import X_from_past_returns, y_from_next_week_return
from models.mlp_returns import fit_predict as mlp_returns_fit_predict

# markets_chosen = ["dow30"]
markets_chosen = ['commodities', 'bonds']
# markets_chosen = ['bonds']
out_name = 'comm_bonds'

preds_df, ports_df = run_walk_forward(
    markets=markets_chosen,
    model_fit_predict_fn=mlp_returns_fit_predict,
    X_fn=lambda w: X_from_past_returns(w, lookback_days=25),
    y_fn=y_from_next_week_return,
    lookback_weeks=5,
    train_weeks=52,
    error_weeks=20,
    start_date="2022-01-01",
    end_date="2025-11-30",
    kappa=1,
    delta=3,
    out_preds_csv=f"outputs/preds_{out_name}_mlp_returns.csv",
    out_ports_csv=f"outputs/ports_{out_name}_mlp_returns.csv",
    out_ports_summary_csv=f"outputs/ports_summary_{out_name}_mlp_returns.csv",
)
print(preds_df.head())
print(ports_df.head())