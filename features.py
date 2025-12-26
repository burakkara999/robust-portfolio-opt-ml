
import pandas as pd
import numpy as np

from old.old_stock_data_module import read_close_prices

def make_weekly_windows(close_prices: pd.DataFrame, lookback=10, horizon=1, days_per_week=5):
    """
    Create rolling weekly windows from daily close prices.

    Returns: list[dict] where each dict contains:
    - t0, t1: timestamps for the window end and horizon end
    - past_prices: (lookback*5 days) x N assets     (X candidate)
    - future_prices: (horizon*5 days) x N assets
    - past_returns: daily log-returns over past window  (X candidate)
    - future_returns: daily log-returns over future window 
    - y_dir : direction labels (assets,)    (y candidate)
    - y_ret : next-week returns (assets,)   (y candidate)
    """
    if close_prices.index.name is None:
        close_prices = close_prices.copy()
        close_prices.index.name = "Date"

    df = close_prices.sort_index().copy()

    # # basic cleaning: forward-fill small gaps, then drop assets with too many NaNs
    # df = df.ffill(limit=2)
    # valid_assets = df.columns[df.notna().mean() >= 0.9]
    # df = df[valid_assets]

    # need enough rows
    L = lookback * days_per_week
    H = horizon * days_per_week
    if len(df) < L + H + 1:
        return []

    windows = []
    step = H  # rebalance every horizon week (usually 5 days)

    for start in range(0, len(df) - (L + H) + 1, step):
        past_prices = df.iloc[start : start + L]
        future_prices = df.iloc[start + L : start + L + H]

        # if any remaining NaNs (e.g., IPO mid-window), skip
        if past_prices.isna().any().any() or future_prices.isna().any().any():
            continue

        past_returns = np.log(past_prices / past_prices.shift(1)).dropna()
        future_returns = np.log(future_prices / future_prices.shift(1)).dropna()

        next_week_logret = future_returns.sum(axis=0)  # per asset
        y_dir = (next_week_logret > 0).astype(int)
        y_ret = next_week_logret  # regression target

        windows.append(
            dict(
                t0=past_prices.index[-1],
                t1=future_prices.index[-1],
                past_prices=past_prices,
                future_prices=future_prices,
                past_returns=past_returns,
                future_returns=future_returns,
                y_dir=y_dir,
                y_ret=y_ret
            )
        )

    return windows


def _build_features(window_prices: pd.DataFrame, days_per_week: int = 5, zscore_cross_section: bool = True):
    """
    window_prices: (L days) x N assets daily closes

    Returns:
      features_df(X): N assets x K features
    """
    prices = window_prices.copy()
    rets = np.log(prices / prices.shift(1)).dropna()

    # helpers
    def cumret(k_days: int):
        # log cumulative return over last k days
        return rets.tail(k_days).sum(axis=0)

    def vol(k_days: int):
        return rets.tail(k_days).std(axis=0)

    def meanret(k_days: int):
        return rets.tail(k_days).mean(axis=0)

    last_ret = rets.tail(1).iloc[0]

    # momentum horizons
    mom_1w  = cumret(1 * days_per_week)
    mom_4w  = cumret(4 * days_per_week)
    mom_10w = cumret(10 * days_per_week) if len(rets) >= 10 * days_per_week else cumret(len(rets))

    # risk / stats
    vol_1w  = vol(1 * days_per_week)
    vol_4w  = vol(4 * days_per_week) if len(rets) >= 4 * days_per_week else vol(len(rets))

    mu_1w   = meanret(1 * days_per_week)
    mu_4w   = meanret(4 * days_per_week) if len(rets) >= 4 * days_per_week else meanret(len(rets))

    # simple max drawdown over lookback
    # drawdown from running max in price level
    running_max = prices.cummax()
    dd = (prices / running_max - 1.0).min(axis=0)  # most negative drawdown

    feats = pd.DataFrame(
        {
            "last_ret": last_ret,
            "mom_1w": mom_1w,
            "mom_4w": mom_4w,
            "mom_10w": mom_10w,
            "vol_1w": vol_1w,
            "vol_4w": vol_4w,
            "mu_1w": mu_1w,
            "mu_4w": mu_4w,
            "max_drawdown": dd,
        }
    )

    # optional: cross-sectional normalization (makes models happier)
    if zscore_cross_section:
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-12)

    return feats


def _make_direction_labels(future_prices: pd.DataFrame):
    future_rets = np.log(future_prices / future_prices.shift(1)).dropna()
    next_week_logret = future_rets.sum(axis=0)
    y_dir = (next_week_logret > 0).astype(int)   # 1 if up, 0 if down
    return y_dir


def _make_return_labels(future_prices: pd.DataFrame):
    future_rets = np.log(future_prices / future_prices.shift(1)).dropna()
    y_ret = future_rets.sum(axis=0)  # next-week log return
    return y_ret


def make_feature_windows(
    close_prices: pd.DataFrame, lookback=10, horizon=1, days_per_week=5):
    """
    Returns a list of dicts:
      - t0, t1
      - X : features (assets x features)
      - y_dir : direction labels (assets,)
      - y_ret : next-week returns (assets,)
    """
    price_windows = make_weekly_windows(
        close_prices,
        lookback=lookback,
        horizon=horizon,
        days_per_week=days_per_week,
    )

    feature_windows = []

    for w in price_windows:
        X = _build_features(w["past_prices"], days_per_week)

        # labels
        y_dir = _make_direction_labels(w["future_prices"])
        y_ret = _make_return_labels(w["future_prices"])

        # align assets (important!)
        common_assets = X.index.intersection(y_dir.index)

        feature_windows.append(
            dict(
                t0=w["t0"],
                t1=w["t1"],
                X=X.loc[common_assets],
                y_dir=y_dir.loc[common_assets],
                y_ret=y_ret.loc[common_assets],
            )
        )

    return feature_windows



# tickers, close_df = read_close_prices("dow30")
# print(close_df.shape)
# windows = make_weekly_windows(close_df)


# print(windows[0].keys())
# print(windows[0]['t0'], windows[0]['t1'])
# print(type(windows[0]['future_returns']))


# y_dir = make_direction_labels(windows[19]['future_prices'])
# print(y_dir)

# y_ret = make_return_labels(windows[19]['future_prices'])
# print(y_ret)

# print(windows[19]['future_returns'])
