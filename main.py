###Import packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime

###Import portfolio construction methods
from risk_parity import weights_risk_parity

###Data
tickers = ['GOOGL', 'AAPL', 'AMZN']
start_date = datetime.datetime(2016, 10, 31)
end_date = datetime.datetime(2017, 10, 31)

###Running methods
w_rp = weights_risk_parity(tickers, start_date, end_date)
# print(w_rp)

###testing area:
