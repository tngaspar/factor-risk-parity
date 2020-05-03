import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import


def get_prices(tickers, start_date, end_date):
    prices = pd.DataFrame([web.DataReader(t, 'yahoo', start_date, end_date).loc[:, 'Adj Close'] for t in tickers],
                          index=tickers).T.asfreq('B').ffill()
    return prices


def get_covariance_matrix(prices):
    covariance_matrix = prices.shape[0] * prices.asfreq('B').pct_change().iloc[1:, :].cov().values

    return covariance_matrix
