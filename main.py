###Import packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime
###Import portfolio construction methods

###Import or download data

###Call applications

###testing area:

tickers=['GOOGL', 'AAPL', 'AMZN']
start_date = datetime.datetime(2016, 10, 31)
end_date = datetime.datetime(2017, 10, 31)

prices = pd.DataFrame([web.DataReader(t, 'yahoo', start_date, end_date).loc[:, 'Adj Close'] for t in tickers],\
                      index=tickers).T.asfreq('B').ffill()



covariances = prices.shape[0] * prices.asfreq('B').pct_change().iloc[1:, :].cov().values

b = np.full(len(tickers), 1/len(tickers))

w1= rp.vanilla.design(covariances,b)
print(w1)
