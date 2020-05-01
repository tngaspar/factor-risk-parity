import numpy as np
import pandas as pd
import riskparityportfolio as rp
from dateutil.relativedelta import relativedelta
import stock_data
from alive_progress import alive_bar
from alive_progress import config_handler
config_handler.set_global(force_tty=True)

def weights_risk_parity(tickers, start_date, end_date):
    prices = stock_data.get_prices(tickers, start_date, end_date)
    cov_matrix = stock_data.get_covariance_matrix(prices)
    budget = np.full(len(tickers), 1 / len(tickers))  # parity of risk budget
    w = rp.vanilla.design(cov_matrix, budget)

    return w


def portfolio_weights_risk_parity(tickers, start_date, end_date, portfolio_update_period):
    business_days_end_months = pd.date_range(start_date, end_date, freq='BM')
    portfolio_weights = pd.DataFrame(index=business_days_end_months, columns=tickers)

    with alive_bar(len(business_days_end_months)) as bar:
        for t in business_days_end_months:
            portfolio_weights.loc[t] = weights_risk_parity(tickers, t + relativedelta(months=-3), t)
            bar()

    return portfolio_weights
