###Import packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime as dt
# from backtester import portfolio_construction
import stock_data

# Import portfolio construction methods
import risk_parity
import factor_risk_parity

# Data
tickers_t = ['GOOGL', 'AAPL', 'AMZN']
start_date_t = dt.datetime(2015, 10, 31)
end_date_t = dt.datetime(2017, 10, 31)

# Running methods
# w_rp = weights_risk_parity(end_date, start_date, tickers)
# print(w_rp)

# testing area:


#def portfolio_construction(method, tickers, start_date, end_date, update_period):
tickers = tickers_t
start_date = start_date_t
end_date = end_date_t

prices = stock_data.get_prices(tickers, start_date, end_date)
#portfolio weights for last business day of month
portfolio = pd.DataFrame([prices.resample('BM', convention='end').asfreq().index], index=['Date']).T
portfolio['weights'] = portfolio['Date'].apply(risk_parity.weights_risk_parity, axis='columns', raw=False, result_type='expand', args=(end_date-dt.timedelta(days=365), tickers))




#.apply(func=risk_parity.weights_risk_parity_backtest, result_type='expand')


#pt = portfolio_construction(1, tickers_t, start_date_t, end_date_t, 'BM')
