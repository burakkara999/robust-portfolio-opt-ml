
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from stock_data_module import read_close_prices_all_merged, read_close_prices
from features import make_weekly_windows, make_feature_windows



markets = ['bist100', 'dow30', 'commodities', 'bonds', 'funds_mini']

results_by_market = {}
models_by_market = {}

for market in markets:
    print(f"\n===== Training model for {market} =====")

    tickers, close_df = read_close_prices_all_merged([market])

    rolling_windows = make_weekly_windows(
        close_prices=close_df,
        lookback=5
    )
    