import pandas_datareader as pdr
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

plt.close('all')

aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2006, 10, 1),
                          end=datetime.datetime(2012, 1, 1))

#aapl.head()
#aapl.tail()
#aapl.describe()

aapl.to_csv('Data/test_aapl.csv')
df=pd.read_csv('Data/test_aapl.csv', header=0, index_col= 'Date', parse_dates=True)

#aapl.index
#aapl.columns
ts = aapl['Close'][-10:]
#type(ts)

#print(aapl.loc[pd.Timestamp('2006-11-01'):pd.Timestamp('2006-12-31')].head)
#print(aapl.loc['2007'].head)
#print(aapl.iloc[22:43])
#print(aapl.iloc[[22,43], [0, 3]])

sample = aapl.sample(20)
#print(sample)

monthly_aapl = aapl.resample('M').mean()
#print(monthly_aapl)
#aapl.asfreq("M", method="bfill")

#ADDING A COLUMN diff
aapl['diff'] = aapl.Open - aapl.Close
#AND DELETING
#del aapl['diff']

#####VISUALIZATION OF TIME-SERIES########

aapl['Close'].plot(grid=True)
#plt.show()

#####SIMPLE FINANCIAL ANALYSIS#######
daily_close = aapl[['Adj Close']]
daily_pct_change = daily_close.pct_change()
#replacing NA with 0:
daily_pct_change.fillna(0, inplace=True)

daily_log_returns = np.log(daily_close.pct_change()+1)

#montly and quarterly returns:
#using last observation as val
monthly = aapl.resample('BM').apply(lambda x: x[-1])
#monthly.pct_change()
quarter = aapl.resample('4M').mean()
#quarter.pct_change()

daily_pct_change_shift = daily_close / daily_close.shift(1) -1

daily_log_returns_shift = np.log(daily_close / daily_close.shift(1))

#######
daily_pct_change.hist(bins=50)
#plt.show()
#print(daily_pct_change.describe())

#####
cum_daily_return = (1 + daily_pct_change).cumprod()
cum_daily_return.plot(figsize=(12,8))

cum_monthly_return = cum_daily_return.resample('M').mean()
cum_monthly_return.plot()

#####GETTING MORE DATA########

def get(tickers, startdate, enddate):
   def data(ticker):
     return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
   datas = map (data, tickers)
   return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

daily_pct_change = daily_close_px.pct_change()

daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1, figsize=(12,12))

#####ROLLING WINDOWS######
adj_close_px = aapl['Adj Close']
moving_avg = adj_close_px.rolling(window=40).mean()
#instead of mean() also moving max(), var(), median() etc.. :
# https://pandas.pydata.org/pandas-docs/version/0.17.0/api.html#standard-moving-window-functions
#print(moving_avg[-10:])

#short moving window
aapl['42'] = adj_close_px.rolling(window=42).mean()
# Long moving window rolling mean
aapl['252'] = adj_close_px.rolling(window=252).mean()
# Plot the adjusted closing price, the short and long windows of rolling means
aapl[['Adj Close', '42', '252']].plot()


###########VOLATILITY CALCULATION ######

min_periods = 75
#volatility calculation
vol = daily_pct_change.rolling(min_periods).std()*np.sqrt(min_periods)
#for the historical one:
#pd.rolling_std(data, window=x) * math.sqrt(window)
vol.plot(figsize=(10,8))


######OLS#########

all_adj_close = all_data[['Adj Close']]
all_returns = np.log(all_adj_close / all_adj_close.shift(1))

# Isolate the AAPL returns
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

# Isolate the MSFT returns
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AAPL and MSFT returns
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']

# Add a constant
X = sm.add_constant(return_data['AAPL'])

# Construct the model
model = sm.OLS(return_data['MSFT'],X).fit()

# Print the summary
#print(model.summary())
#return_data.info()

#plot returns
# plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
# #add axix
# ax = plt.axis()
#
# x = np.linspace(ax[0], ax[1] + 0.01)
#
# # Plot the regression line
# plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
#
# # Customize the plot
# plt.grid(True)
# plt.axis('tight')
# plt.xlabel('Apple Returns')
# plt.ylabel('Microsoft returns')
#
# # Show the plot
# #plt.show()
#
# # Plot the rolling correlation
# return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()

# Show the plot
#plt.show()

##SIMPLE EXAMPLE TRADING STRAT##
short_window = 40
long_window = 100
#signals dataframe
signals = pd.DataFrame(index=aapl.index)
signals['signal']= 0.0

signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

signals['positions'] = signals['signal'].diff()

# initialize fig
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Price in $')
aapl['Close'].plot(ax=ax1, color='r', lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')

# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')

####Backtesting####
initial_capital = float(100_000.0)
positions = pd.DataFrame(index=signals.index).fillna(0.0)
#buy 100 shares
positions['AAPL'] = 100*signals['signal']
portfolio = positions.multiply(aapl['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)
ax1.plot(portfolio.loc[signals.positions == 1.0].index,
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index,
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')

###Evaluating performance###
returns = portfolio['returns']
# annualized sharp ratio
sharpe_ratio = np.sqrt(252)*(returns.mean() / returns.std())

# maximum drawdown (on the last year)
window = 252
rolling_max = aapl['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = aapl['Adj Close']/rolling_max - 1.0
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
daily_drawdown.plot()
max_daily_drawdown.plot()

# Compound Annual Growth Rate (CAGR)
# Get the number of days in `aapl`
days = (aapl.index[-1] - aapl.index[0]).days
# Calculate the CAGR
cagr = ((((aapl['Adj Close'][-1]) / aapl['Adj Close'][1])) ** (365.0/days)) - 1