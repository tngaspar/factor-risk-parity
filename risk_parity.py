import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime
import stock_data


def weights_risk_parity(tickers, start_date, end_date):
    prices = stock_data.get_prices(tickers, start_date, end_date)
    cov_matrix = stock_data.get_covariance_matrix(prices)
    budget = np.full(len(tickers), 1 / len(tickers))  # parity of risk budget
    w = rp.vanilla.design(cov_matrix, budget)

    return w


def portfolio_weights_risk_parity(tickers, start_date, end_date, portfolio_update_period):

    #bussiness_days_rng = pd.date_range(start, end, freq='BM')
    w=1

    return w