
from stock_data_module import read_close_prices_all_merged
from features import make_weekly_windows, make_feature_windows

tickers, close_df = read_close_prices_all_merged(['bist100', 'dow30', 'commodities', 'bonds', 'funds_mini'])
print(close_df.shape)



rolling_windows = make_weekly_windows(close_prices=close_df, lookback=10) ## lookback weeks!

rolling_windows_features = make_feature_windows(close_prices=close_df, lookback=5)

print(rolling_windows[0])

