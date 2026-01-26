#!/usr/bin/env python3
import os
import pandas as pd

# Folder where THIS script lives (not where you run it from)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df.columns = df.columns.astype(str)
    return df.sort_index()


def merge_predictions(
    model_name: str,
    markets: list[str],
    out_markets: str | None = None,
    join: str = "inner",          # inner: intersection of periods; outer: union
    prefix_cols: bool = False,    # prefix columns with "market__"
):
    if out_markets is None:
        out_markets = "-".join(markets)

    kinds = {"expected_returns": [], "true_returns": [], "errors": []}

    # read
    for mkt in markets:
        for kind in kinds.keys():
            filename = f"{model_name}_{mkt}_{kind}.csv"
            path = os.path.join(SCRIPT_DIR, filename)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            df = _read_wide_csv(path)

            if prefix_cols:
                df = df.copy()
                df.columns = [f"{mkt}__{c}" for c in df.columns]

            kinds[kind].append(df)

    if join not in ("inner", "outer"):
        raise ValueError("join must be 'inner' or 'outer'")

    def align_and_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if join == "inner":
            idx = dfs[0].index
            for d in dfs[1:]:
                idx = idx.intersection(d.index)
            dfs = [d.loc[idx] for d in dfs]
            merged = pd.concat(dfs, axis=1)
        else:
            merged = pd.concat(dfs, axis=1)  # outer align by default

        merged = merged.apply(pd.to_numeric, errors="coerce")
        merged = merged.sort_index()
        merged = merged.reindex(sorted(merged.columns), axis=1)
        return merged

    expected_df = align_and_concat(kinds["expected_returns"])
    true_df     = align_and_concat(kinds["true_returns"])
    errors_df   = align_and_concat(kinds["errors"])

    out_paths = {
        "expected_returns": os.path.join(SCRIPT_DIR, f"{model_name}_{out_markets}_merged_expected_returns.csv"),
        "true_returns":     os.path.join(SCRIPT_DIR, f"{model_name}_{out_markets}_merged_true_returns.csv"),
        "errors":           os.path.join(SCRIPT_DIR, f"{model_name}_{out_markets}_merged_errors.csv"),
    }

    expected_df.to_csv(out_paths["expected_returns"])
    true_df.to_csv(out_paths["true_returns"])
    errors_df.to_csv(out_paths["errors"])

    print("Saved merged tables:")
    print(" ", out_paths["expected_returns"], expected_df.shape)
    print(" ", out_paths["true_returns"], true_df.shape)
    print(" ", out_paths["errors"], errors_df.shape)

    return expected_df, true_df, errors_df



merge_predictions(
    model_name="MLP",
    markets=["dow30", "commodities", "bonds"],
    join="inner",
    prefix_cols=False,
)