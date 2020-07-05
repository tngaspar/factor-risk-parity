# functions to use after obtaining portfolio weights over the back-test period

import stock_data
import pandas as pd


def daily_returns_of_portfolio(weights_portfolio):
    """ Calculates the daily returns of a portfolio

    :param weights_portfolio: The weights of the portfolio's equities over the back-test period
    :return: daily returns of the portfolio
    """
    tickers = weights_portfolio.columns
    start_date = weights_portfolio.index[0]
    end_date = weights_portfolio.index[-1]
    stock_returns = stock_data.get_daily_returns(tickers, start_date, end_date)[1:]

    daily_returns = pd.Series(
        (stock_returns * (weights_portfolio.asfreq('B').ffill().shift(1))).sum(1),
        index=pd.to_datetime(stock_returns.index), name='Returns')

    return daily_returns


def cumulative_returns(returns):
    """ Calculates cumulative returns

    :param returns: returns during a period
    :return: cummulative returns
    """
    cum_ret = (returns + 1).cumprod()

    return cum_ret


