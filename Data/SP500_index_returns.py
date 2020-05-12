import pandas as pd
from stock_data import get_prices_yf
import datetime as dt

start_date = dt.date(2006, 12, 30)
end_date = dt.date(2019, 12, 30)

prices = get_prices_yf(['^GSPC'], start_date, end_date)
SP500_returns = prices.asfreq('B').pct_change().rename(columns={'^GSPC': 'SP_500'})[1:]
SP500_returns.to_csv('SP500_index_daily_returns.csv')