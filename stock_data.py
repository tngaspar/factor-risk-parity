from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
yf.pdr_override()


def get_prices_yf(tickers, start_date, end_date):
    prices = pd.DataFrame([pdr.get_data_yahoo(t, start_date, end_date).loc[:, 'Adj Close'] for t in tickers],
                          index=tickers).T.asfreq('B').ffill()
    return prices


def get_prices_file(tickers, start_date, end_date):
    data = pd.read_csv('Data\Equities\SP500.csv', parse_dates=True, index_col=0)
    prices = pd.DataFrame([data.loc[start_date:end_date, t] for t in tickers]
                          ).T.ffill()

    return prices


def get_prices(tickers, start_date, end_date):
    prices = get_prices_file(tickers, start_date, end_date)
    # prices = (tickers, start_date, end_date)
    return prices


def get_daily_returns(tickers, start_date, end_date):
    returns = get_prices(tickers, start_date, end_date).asfreq('B').pct_change()

    return returns


def get_covariance_matrix(prices):
    covariance_matrix = prices.shape[0] * prices.asfreq('B').pct_change().iloc[1:, :].cov().values

    return covariance_matrix


def get_sp500_index_returns(start_date, end_date):
    daily_sp500_ret = pd.DataFrame(pd.read_csv(r'Data/SP500_index_daily_returns.csv')['SP_500'])
    daily_sp500_ret.index = pd.to_datetime(pd.read_csv(r'Data/SP500_index_daily_returns.csv')['Date'])
    daily_sp500_ret = daily_sp500_ret[start_date:end_date]
    return daily_sp500_ret

