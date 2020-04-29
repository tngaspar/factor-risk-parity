import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime
import stock_data


def weights_risk_parity(end_date, start_date, tickers):
    prices = stock_data.get_prices(tickers, start_date, end_date)
    cov_matrix = stock_data.get_covariance_matrix(prices)
    budget = np.full(len(tickers), 1 / len(tickers))  # parity of risk budget
    w = rp.vanilla.design(cov_matrix, budget)

    return w


