import pandas as pd
import yfinance as yf

def download_close_prices_all(data_name="bist100", start_date="2023-01-01", end_date="2025-01-01"):
    """tickers_path = data/{data_name}_tickers.csv --> 1 column "ticker"

    start_date, end_date --> "yyyy-mm-dd"
    
    output_path = data/close_prices/{data_name}_close_prices.csv
    """
    tickers_path = f"data/tickers/{data_name}_tickers.csv"
    out_path = f"data/close_prices/{data_name}_close_prices.csv"
    
    ## Download - close prices
    df_tickers = pd.read_csv(tickers_path)
    tickers = df_tickers["ticker"].to_list()
    
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=False
    )["Close"]

    prices.to_csv(out_path)


def download_close_prices(data_name="bist100", start_date="2023-01-01", end_date="2025-01-01", batch_size=100):
    """tickers_path = data/{data_name}_tickers.csv --> 1 column "ticker"

    start_date, end_date --> "yyyy-mm-dd"
    
    output_path = data/close_prices/{data_name}_close_prices.csv
    """
    tickers_path = f"data/tickers/{data_name}_tickers.csv"
    out_path = f"data/close_prices/{data_name}_close_prices.csv"
    
    ## Download - close prices
    df_tickers = pd.read_csv(tickers_path)
    tickers = df_tickers["ticker"].to_list()
    
    batch = 0
    for slice in range(0,len(tickers),batch_size):
        t_set = tickers[slice:slice+batch_size]
        batch += 1
        out_path = f"data/close_prices/{data_name}_close_prices_b{batch}.csv"
        prices = yf.download(
            t_set,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            group_by="column",
            threads=False
        )["Close"]

        prices.to_csv(out_path)


def read_close_prices_stock(data_name="bist100", after_date=None):
    """tickers_path = data/{data_name}_tickers.csv --> 1 column "ticker"

    close_prices_path = data/close_prices/{data_name}_close_prices.csv

    returns --> tickers: list, close_prices: DataFrame
    """
    
    df_tickers = pd.read_csv(f"data/tickers/{data_name}_tickers.csv")
    tickers = df_tickers["ticker"].to_list()
    
    prices_df = pd.read_csv(f"data/close_prices/{data_name}_close_prices.csv")

    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    prices_df = prices_df.set_index(prices_df["Date"])
    prices_df = prices_df.sort_index()
    prices_df = prices_df.drop(columns=['Date'])
    # print(prices_df.head)
    # print(prices_df.shape)

    ## remove tickers that are not have enough data -- probably new or old in bist100
    min_coverage = 0.9 
    valid_assets = prices_df.columns[prices_df.notna().mean() >= min_coverage]
    prices_df = prices_df[valid_assets]
    print(prices_df.shape)

    prices_df = prices_df.ffill(limit=2) ## fill 1-2 period na values with latest price value
    
    if after_date != None:
        ## after 3 months all prices are valid -- "2022-04-01" for bist100
        prices_df = prices_df.loc[after_date:]

    # for t in valid_assets:
    #     na_counts = prices_after[t].isna().sum()
    #     if na_counts > 0:
    #         print(t, na_counts)
    return tickers, prices_df



# download_close_prices("sp500", batch_size=50)
# download_close_prices_all("dow30")
# tickers, close_df = read_close_prices("dow30")
# print(len(tickers), close_df.shape)
