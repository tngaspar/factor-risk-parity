import pyfolio as pf
import stock_data
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime


def portfolio_construction(method, tickers, start_date, end_date, update_period=str):

    prices = stock_data.get_prices(tickers, start_date, end_date)
    weights = pd.DataFrame(index=prices.index)
    weights = weights.resample(update_period)
    weights = 1

    return weights
