###Import packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime as dt
# from backtester import portfolio_construction
import stock_data
from dateutil.relativedelta import relativedelta
import importlib
import pyfolio as pf

# Import portfolio construction methods
import risk_parity as rp
importlib.reload(rp)
import factor_risk_parity

# Data
tickers = ['GOOGL', 'AAPL', 'AMZN']
start_date = dt.datetime(2015, 10, 31)
end_date = dt.datetime(2017, 10, 31)

# Running methods
# w_rp = rp.weights_risk_parity(tickers, start_date, end_date)
w_pt_rp = rp.portfolio_weights_risk_parity(tickers, start_date, end_date, 'BM')

# testing area:

