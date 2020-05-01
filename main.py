###Import packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime as dt
# from backtester import portfolio_construction
import stock_data
from dateutil.relativedelta import relativedelta
from alive_progress import alive_bar
from alive_progress import config_handler
config_handler.set_global(force_tty=True)

# Import portfolio construction methods
from risk_parity import weights_risk_parity
import factor_risk_parity

# Data
tickers = ['GOOGL', 'AAPL', 'AMZN']
start_date = dt.datetime(2015, 10, 31)
end_date = dt.datetime(2017, 10, 31)

# Running methods
# w_rp = weights_risk_parity(tickers, start_date, end_date)
# print(w_rp)

# testing area:


portfolio_update_frequency = 'BM'
business_days_end_months = pd.date_range(start_date, end_date, freq='BM')
#w = weights_risk_parity(tickers, business_days_end_months[0]+ relativedelta(years=-1), business_days_end_months[0])
portfolio_weights = pd.DataFrame(index=business_days_end_months, columns=tickers)

with alive_bar(len(business_days_end_months)) as bar:
    for t in business_days_end_months:
        portfolio_weights.loc[t] = weights_risk_parity(tickers, t + relativedelta(months=-3), t)
        bar()

#prices = stock_data.get_prices(tickers, start_date, end_date)
#portfolio weights for last business day of month
#portfolio = pd.DataFrame([prices.resample('BM', convention='end').asfreq().index], index=['Date']).T
#portfolio['weights'] = portfolio['Date'].apply(risk_parity.weights_risk_parity, axis='columns', raw=False, result_type='expand', args=(end_date-dt.timedelta(days=365), tickers))




#.apply(func=risk_parity.weights_risk_parity_backtest, result_type='expand')


#pt = portfolio_construction(1, tickers_t, start_date_t, end_date_t, 'BM')
