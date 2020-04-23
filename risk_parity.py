import numpy as np
import pandas as pd
import pandas_datareader.data as web
import riskparityportfolio as rp
import datetime

def weights_risk_parity(tickers, start_date, end_date):

    prices = pd.DataFrame([web.DataReader(t, 'yahoo', start_date, end_date).loc[:, 'Adj Close'] for t in tickers], \
                          index=tickers).T.asfreq('B').ffill()

    covariances = prices.shape[0] * prices.asfreq('B').pct_change().iloc[1:, :].cov().values

    b = np.full(len(tickers), 1 / len(tickers))

    w = rp.vanilla.design(covariances, b)

    return w