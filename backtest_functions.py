import stock_data
import pandas as pd


def daily_returns_of_portfolio(weights_portfolio):
    tickers = weights_portfolio.columns
    start_date = weights_portfolio.index[0]
    end_date = weights_portfolio.index[-1]
    stock_returns = stock_data.get_daily_returns(tickers, start_date, end_date)[1:]

    daily_returns = pd.Series(
        (stock_returns * (weights_portfolio.asfreq('B').ffill().shift(1))).sum(1),
        index=pd.to_datetime(stock_returns.index), name='Returns')

    return daily_returns
