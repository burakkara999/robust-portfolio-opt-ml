import pandas as pd
import yfinance as yf

def download_close_prices_bist100(tickers_path="bist100_tickers.csv",out_path="bist100_close_prices.csv"):
    ## Download bist100 - close prices
    df_tickers = pd.read_csv(tickers_path)
    tickers = df_tickers["CONSTITUENT CODE"].to_list()

    prices = yf.download(
        tickers,
        start="2022-01-01",
        end="2025-01-01",
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True
    )["Close"]

    prices.to_csv(out_path)

def read_close_prices_bist100():
    """returns --> tickers: list, close_prices: DataFrame """
    df_tickers = pd.read_csv("bist100_tickers.csv")
    tickers = df_tickers["CONSTITUENT CODE"].to_list()
    
    prices_df = pd.read_csv('bist100_close_prices.csv')

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
    # print(prices_df.shape)

    prices_df = prices_df.ffill(limit=2) ## fill 1-2 period na values with latest price value
    ## after 3 months all prices are valid
    prices_after = prices_df.loc["2022-04-01":]

    # for t in valid_assets:
    #     na_counts = prices_after[t].isna().sum()
    #     if na_counts > 0:
    #         print(t, na_counts)
    return tickers, prices_after
