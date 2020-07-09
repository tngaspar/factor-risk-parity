import stock_data
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from alive_progress import config_handler
config_handler.set_global(force_tty=True)


def ew_weights(stock_tickers, date):
    stock_returns = stock_data.get_daily_returns(stock_tickers, date, date)
    n_stocks = stock_returns.shape[1]
    x = np.ones(n_stocks) / n_stocks

    return x


def portfolio_weights_risk_parity(tickers, start_date, end_date, portfolio_rebalance_period):
    business_days_end_months = pd.date_range(start_date, end_date, freq=portfolio_rebalance_period)
    portfolio_weights = pd.DataFrame(index=business_days_end_months, columns=tickers)

    with alive_bar(len(business_days_end_months)) as bar:
        for t in business_days_end_months:
            portfolio_weights.loc[t] = ew_weights(tickers, t)
            bar()

    return portfolio_weights

