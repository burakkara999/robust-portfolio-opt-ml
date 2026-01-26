import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


TICKERS_DIR = Path("data/tickers")
PRICES_DIR  = Path("data/close_prices")


# ----------------------------
# Helpers
# ----------------------------
def list_data_names(tickers_dir=TICKERS_DIR):
    """
    Returns list of data_name from files like: {data_name}_tickers.csv
    """
    if not tickers_dir.exists():
        raise FileNotFoundError(f"Missing folder: {tickers_dir}")

    names = []
    for p in tickers_dir.glob("*_tickers.csv"):
        name = p.name.replace("_tickers.csv", "")
        names.append(name)
    return sorted(names)


def read_tickers(data_name, tickers_dir=TICKERS_DIR):
    tickers_path = tickers_dir / f"{data_name}_tickers.csv"
    df = pd.read_csv(tickers_path)

    if "ticker" not in df.columns:
        raise ValueError(f"{tickers_path} must contain a 'ticker' column.")

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return tickers


def _extract_close(prices_raw, tickers):
    """
    yfinance returns:
      - MultiIndex columns if multiple tickers
      - Single-index columns if one ticker
    """
    if isinstance(prices_raw.columns, pd.MultiIndex):
        close = prices_raw["Close"]
    else:
        # single ticker case: "Close" is a Series
        close = prices_raw["Close"].to_frame(name=tickers[0])

    close.index.name = "Date"
    close = close.sort_index()
    close = close.dropna(how="all")
    return close


def save_log_returns(prices_df, out_path, lag=1):

    log_returns = np.log(prices_df / prices_df.shift(lag))

    # keep only every lag-th row 
    log_returns = log_returns.iloc[::lag]

    log_returns = log_returns.dropna(how="all")
    log_returns.to_csv(out_path)
    return out_path


# ----------------------------
# Downloaders
# ----------------------------
def download_close_prices_all(
    data_name,
    start_date="2022-01-01",
    end_date="2025-12-01",
    interval="1d",
    tickers_dir=TICKERS_DIR,
    prices_dir=PRICES_DIR,
):
    """
    Downloads ALL tickers in one shot and saves:
      data/close_prices/{data_name}_close_prices.csv
    """
    prices_dir.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(data_name, tickers_dir=tickers_dir)
    out_path = prices_dir / f"{data_name}_close_prices.csv"

    prices_raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        group_by="column",
        threads=False,
        timeout=300,
    )

    close = _extract_close(prices_raw, tickers)
    close.to_csv(out_path)
    return out_path


def download_close_prices_batched(
    data_name,
    start_date="2022-01-01",
    end_date="2025-12-01",
    interval="1d",
    batch_size=100,
    merge_batches=True,
    tickers_dir=TICKERS_DIR,
    prices_dir=PRICES_DIR,
):
    """
    Downloads tickers in batches and writes:
      data/close_prices/{data_name}_close_prices_b{batch}.csv

    If merge_batches=True, also creates:
      data/close_prices/{data_name}_close_prices.csv
    """
    prices_dir.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(data_name, tickers_dir=tickers_dir)

    batch_files = []
    batch = 0

    for start in range(0, len(tickers), batch_size):
        batch += 1
        t_set = tickers[start : start + batch_size]
        out_path = prices_dir / f"{data_name}_close_prices_b{batch}.csv"

        prices_raw = yf.download(
            tickers=t_set,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            group_by="column",
            threads=False,
            timeout=120,
        )

        close = _extract_close(prices_raw, t_set)
        close.to_csv(out_path)
        batch_files.append(out_path)

    if merge_batches:
        merged_path = merge_close_price_batches(data_name, prices_dir=prices_dir)
        return batch_files, merged_path

    return batch_files, None


def merge_close_price_batches(data_name, prices_dir=PRICES_DIR):
    """
    Reads all files like:
      {data_name}_close_prices_b*.csv
    merges columns on Date index, saves:
      {data_name}_close_prices.csv
    """
    pattern = f"{data_name}_close_prices_b*.csv"
    batch_paths = sorted(prices_dir.glob(pattern))

    if not batch_paths:
        raise FileNotFoundError(f"No batch files found for {data_name}: {pattern}")

    dfs = []
    for p in batch_paths:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        dfs.append(df)

    merged = pd.concat(dfs, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]  # in case overlaps
    merged.index.name = "Date"
    merged = merged.sort_index()

    out_path = prices_dir / f"{data_name}_close_prices.csv"
    merged.to_csv(out_path)
    return out_path


def download_all_datasets(
    start_date="2022-01-01",
    end_date="2025-12-01",
    interval="1d",
    mode="auto",          # "auto" | "all" | "batched"
    batch_size=100,
    tickers_dir=TICKERS_DIR,
    prices_dir=PRICES_DIR,
):
    """
    Downloads for every {data_name}_tickers.csv in data/tickers.
    mode:
      - "all": use single-shot downloader
      - "batched": always use batched downloader (and merge)
      - "auto": if ticker count > batch_size -> batched else all
    """
    results = {}
    for name in list_data_names(tickers_dir):
        tickers = read_tickers(name, tickers_dir=tickers_dir)

        if mode == "all":
            out = download_close_prices_all(name, start_date, end_date, interval, tickers_dir, prices_dir)
            results[name] = {"method": "all", "out": str(out)}
        elif mode == "batched":
            _, out = download_close_prices_batched(name, start_date, end_date, interval, batch_size, True, tickers_dir, prices_dir)
            results[name] = {"method": "batched", "out": str(out)}
        else:  # auto
            if len(tickers) > batch_size:
                _, out = download_close_prices_batched(name, start_date, end_date, interval, batch_size, True, tickers_dir, prices_dir)
                results[name] = {"method": "batched", "out": str(out)}
            else:
                out = download_close_prices_all(name, start_date, end_date, interval, tickers_dir, prices_dir)
                results[name] = {"method": "all", "out": str(out)}
    return results


# ----------------------------
# Readers
# ----------------------------

    """
    Reads:
      data/close_prices/{data_name}_close_prices.csv

    Returns:
      tickers (from ticker file), prices_df (filtered & cleaned)
    """
def read_close_prices(
    data_name,
    after_date=None,
    min_coverage=0.9,
    ffill_limit=2,
    ipo_pad=True,         # <--- add this
    tickers_dir=TICKERS_DIR,
    prices_dir=PRICES_DIR,
):
    """
    Reads:
        data/close_prices/{data_name}_close_prices.csv

    ipo_pad=True:
        Fill leading NaNs (before first observed price) with the first observed price.
        This produces ~0 returns before listing (flat pre-history).
    
    Returns:
        tickers (from ticker file), prices_df (filtered & cleaned)  
    """
    tickers = read_tickers(data_name, tickers_dir=tickers_dir)

    prices_path = prices_dir / f"{data_name}_close_prices.csv"
    prices_df = pd.read_csv(prices_path)

    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    prices_df = prices_df.set_index("Date").sort_index()

    if after_date is not None:
        prices_df = prices_df.loc[pd.to_datetime(after_date):]

    # --- IPO padding (flat pre-history) ---
    if ipo_pad: ## for new ones
        # bfill fills leading NaNs with the first non-NaN value in that column
        prices_df = prices_df.bfill()

    # forward-fill small internal gaps (e.g., holidays / short missing streaks)
    if ffill_limit is not None and ffill_limit > 0:
        prices_df = prices_df.ffill(limit=ffill_limit)

    # keep only assets with sufficient non-NA coverage (after fills)
    valid_assets = prices_df.columns[prices_df.notna().mean() >= min_coverage]
    prices_df = prices_df[valid_assets]

    return tickers, prices_df


def read_close_prices_all_merged(
    data_names=None,       # None => detect from tickers folder
    after_date=None,
    min_coverage=0.9,
    ffill_limit=2,
    how="inner",           # "inner" (common dates) or "outer"
    tickers_dir=TICKERS_DIR,
    prices_dir=PRICES_DIR,
    add_prefix=True,       # avoid column name collisions across datasets
):
    """
    Reads + merges all datasets into one DataFrame (Date index).
    Returns:
      all_tickers: dict[data_name -> list]
      merged_prices_df: DataFrame
    """
    if data_names is None:
        data_names = list_data_names(tickers_dir)

    all_tickers = {}
    dfs = []

    for name in data_names:
        tickers, df = read_close_prices(
            name,
            after_date=after_date,
            min_coverage=min_coverage,
            ffill_limit=ffill_limit,
            tickers_dir=tickers_dir,
            prices_dir=prices_dir,
        )
        all_tickers[name] = tickers

        if add_prefix:
            df = df.copy()
            df.columns = [f"{name}:{c}" for c in df.columns]

        dfs.append(df)

    if not dfs:
        return all_tickers, pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how=how)

    merged = merged.sort_index()
    return all_tickers, merged


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # 1) Download everything found in data/tickers/
    # results = download_all_datasets(mode="auto", batch_size=100, end_date="2026-01-01")
    # print(results)

    # # 2) Read single dataset
    # tickers, prices = read_close_prices("bist100", after_date="2023-06-01")
    # print(len(tickers), prices.shape)

    # # 3) Read + merge all datasets
    # all_tickers, merged_prices = read_close_prices_all_merged(after_date="2023-06-01", how="inner")
    # print(merged_prices.shape)
    pass


##TEST CODE###
# print(list_data_names())
# download_close_prices_all('funds_mini')

# # download_close_prices_batched('bist100', batch_size=20)
# tickers1, close_df1 = read_close_prices('bist100')
# print(close_df1.shape)

# tickers2, close_df2 = read_close_prices('dow30')
# print(close_df2.shape)

# tickers3, close_df3 = read_close_prices('commodities')
# print(close_df3.shape)

# tickers4, close_df4 = read_close_prices('bonds')
# print(close_df4.shape)

# tickers5, close_df5 = read_close_prices('funds_mini')
# print(close_df5.shape)

# tickers, close_df = read_close_prices_all_merged(['bist100', 'dow30', 'commodities', 'bonds', 'funds_mini'])
# print(close_df.shape)

