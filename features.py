
import pandas as pd
import numpy as np

# from old.old_stock_data_module import read_close_prices

def make_weekly_windows(close_prices: pd.DataFrame, lookback=10, horizon=1, days_per_week=5):
    """
    Create rolling weekly windows from daily close prices.

    Returns: list[dict] where each dict contains:
    - t0, t1: timestamps for the window end and horizon end
    - past_prices: (lookback*5 days) x N assets     (X candidate)
    - future_prices: (horizon*5 days) x N assets
    - past_returns: daily log-returns over past window  (X candidate)
    - future_returns: daily log-returns over future window 
    - past_weekly_ret: weekly log-returns over past window (X candidate)
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

        past_weekly_returns = past_returns.groupby(np.arange(len(past_returns)) // days_per_week).sum()
        # past_weekly_ret shape: (lookback, assets)  # each row is one week log-return

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
                past_weekly_returns=past_weekly_returns,
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
    # Use fillna(0) cautiously or handle NaNs downstream; 
    # here we assume window_prices is clean enough or we accept data loss.
    rets = np.log(prices / prices.shift(1)).fillna(0) 

    # --- Helpers ---
    def cumret(k_days: int):
        return rets.tail(k_days).sum(axis=0)

    def vol(k_days: int):
        # Add small epsilon to avoid division by zero
        return rets.tail(k_days).std(axis=0) + 1e-8 
    
    # --- 1. Momentum (Raw) ---
    mom_1w  = cumret(1 * days_per_week)
    mom_4w  = cumret(4 * days_per_week)
    mom_12w = cumret(12 * days_per_week) # Changed 10->12 (Quarterly is a common seasonal frequency)

    # --- 2. Volatility (Risk) ---
    vol_1w  = vol(1 * days_per_week)
    vol_4w  = vol(4 * days_per_week)
    
    # --- 3. Risk-Adjusted Momentum (New High-Value Feature) ---
    # Captures "Smoothness" of the trend. 
    # High return with low vol = High Signal.
    sharpe_1w = mom_1w / vol_1w
    sharpe_4w = mom_4w / vol_4w

    # --- 4. Drawdown ---
    running_max = prices.cummax()
    dd = (prices / running_max - 1.0).min(axis=0)

    # --- 5. Volatility Regime (New) ---
    # Is current vol higher than long-term vol? (Mean Reversion signal)
    vol_ratio = vol_1w / (vol(12 * days_per_week) + 1e-8)

    feats = pd.DataFrame(
        {
            "mom_1w": mom_1w,
            "mom_4w": mom_4w,
            "mom_12w": mom_12w,
            "vol_1w": vol_1w,
            "vol_4w": vol_4w,
            "sharpe_1w": sharpe_1w, # New
            "sharpe_4w": sharpe_4w, # New
            "vol_ratio": vol_ratio, # New
            "max_drawdown": dd,
        }
    )
    
    # --- 6. Cross-Sectional Normalization (Crucial) ---
    if zscore_cross_section:
        # Standardize across ASSETS (axis=0 of the feats DF)
        # Result: For each feature, mean is 0, std is 1 across all stocks for that day.
        # This forces the model to learn relative ranking instead of absolute price levels.
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
        
    return feats

def _build_features_advanced(window_prices: pd.DataFrame, days_per_week: int = 5, zscore_cross_section: bool = True):
    prices = window_prices.copy()
    rets = np.log(prices / prices.shift(1)).fillna(0)

    # --- Helpers ---
    def cumret(k_days): return rets.tail(k_days).sum(axis=0)
    def vol(k_days): return rets.tail(k_days).std(axis=0) + 1e-8

    # --- New Helper: RSI ---
    def calc_rsi(series, window=14):
        # Vectorized RSI calculation
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    # --- 1. Momentum & Volatility (Existing) ---
    mom_4w  = cumret(4 * days_per_week)
    vol_4w  = vol(4 * days_per_week)
    sharpe_4w = mom_4w / vol_4w

    # --- 2. Mean Reversion (New: RSI) ---
    # We use the full price history to calculate the latest RSI value
    # RSI is typically 14 days
    rsi_val = calc_rsi(prices, window=14).iloc[-1]
    
    # Scale RSI to 0-1 range for better MLP convergence (standard is 0-100)
    rsi_scaled = rsi_val / 100.0 

    # --- 3. Trend Stability (New: Distance to MA) ---
    # Is the price extended far above its trend?
    ma_50 = prices.rolling(window=50).mean().iloc[-1]
    curr_price = prices.iloc[-1]
    
    # "Distance": (Price / MA) - 1. 
    # Positive = Above trend, Negative = Below trend.
    dist_ma_50 = (curr_price / (ma_50 + 1e-8)) - 1.0

    # --- 4. Assemble ---
    feats = pd.DataFrame({
        "mom_4w": mom_4w,
        "vol_4w": vol_4w,
        "sharpe_4w": sharpe_4w,
        "rsi": rsi_scaled,       # <--- New
        "dist_ma_50": dist_ma_50 # <--- New
    })

    # --- 5. Cross-Sectional Normalization ---
    if zscore_cross_section:
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

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

   
def make_feature_windows(close_prices: pd.DataFrame, lookback=10, horizon=1, days_per_week=5):
    """
    Returns: list[dict], where each dict contains:
    - t0, t1: timestamps for the window end and horizon end
    - past_prices: (lookback*5 days) x N assets     (X candidate)
    - future_prices: (horizon*5 days) x N assets
    - past_returns: daily log-returns over past window  (X candidate)
    - future_returns: daily log-returns over future window 
    - past_weekly_ret: weekly log-returns over past window (X candidate)
    - y_dir : direction labels (assets,)    (y candidate)
    - y_ret : next-week returns (assets,)   (y candidate)

    Plus:
    - X_feat: features (assets x features)
    """
    price_windows = make_weekly_windows(
        close_prices,
        lookback=lookback,
        horizon=horizon,
        days_per_week=days_per_week,
    )

    feature_windows = []

    for w in price_windows:
        # Build features from past window (you can also use w["past_returns"] if your feature builder prefers)
        X = _build_features(w["past_prices"], days_per_week)
        # X = _build_features_advanced(w["past_prices"], days_per_week)

        # Use labels already computed in make_weekly_windows
        y_dir = w["y_dir"]
        y_ret = w["y_ret"]

        # ---- Align assets across everything we carry forward ----
        # past/future prices are DataFrames with columns=assets
        common_assets = X.index
        common_assets = common_assets.intersection(w["past_prices"].columns)
        common_assets = common_assets.intersection(w["future_prices"].columns)
        common_assets = common_assets.intersection(w["past_returns"].columns)
        common_assets = common_assets.intersection(w["future_returns"].columns)

        # y_* can be Series with index=assets (or 1-row df). Handle both safely:
        if hasattr(y_dir, "index"):
            common_assets = common_assets.intersection(y_dir.index)
        if hasattr(y_ret, "index"):
            common_assets = common_assets.intersection(y_ret.index)

        common_assets = common_assets.sort_values()

        # Slice everything consistently
        out = dict(w)  # keep all original keys/values
        out["X_feat"] = X.loc[common_assets]

        out["past_prices"] = w["past_prices"][common_assets]
        out["future_prices"] = w["future_prices"][common_assets]
        out["past_returns"] = w["past_returns"][common_assets]
        out["future_returns"] = w["future_returns"][common_assets]

        # labels
        out["y_dir"] = y_dir.loc[common_assets] if hasattr(y_dir, "loc") else y_dir
        out["y_ret"] = y_ret.loc[common_assets] if hasattr(y_ret, "loc") else y_ret

        feature_windows.append(out)

    return feature_windows



from asset_data_module import read_close_prices_all_merged

tickers, close_df = read_close_prices_all_merged(["dow30"])  ## 980days x N assets

##191 weeks = 980/5 - lookback*5
rolling_feature_windows = make_feature_windows(close_prices=close_df, lookback=5) ## 191weeks --> X: N assets x 9 features


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
