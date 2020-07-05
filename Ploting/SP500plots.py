# Simple analysis of the S&P500 index for thesis writting

import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import empyrical as ep
import matplotlib.pylab as pylab
sns.set(style="white", context='paper')
params = {'axes.titlesize':'large'}
pylab.rcParams.update(params)



daily_sp500_ret = pd.DataFrame(pd.read_csv(r'../Data/SP500_index_daily_returns.csv')['SP_500'])
daily_sp500_ret.index = pd.to_datetime(pd.read_csv(r'../Data/SP500_index_daily_returns.csv')['Date'])
daily_sp500_ret = daily_sp500_ret[dt.date(1990, 1, 1):]


sp500 = daily_sp500_ret.rename(columns={'SP_500': 'Returns'})
sp500['Cumulative Returns'] = (sp500['Returns'] + 1).cumprod()

sns.lineplot(y='Cumulative Returns', x=sp500.index, data=sp500, ci=None, estimator=None)
plt.title('Cumulative returns of the S&P 500')
plt.ylabel('Cumulative Returns')
plt.xlabel('')
plt.grid(True)
plt.ylim(0, 10)
plt.xlim(dt.datetime(1990,1 , 1), dt.datetime(2021,1,1))
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, lw=2)
plt.savefig('SP500_cum_ret.pdf')
plt.show()
plt.close()



sp500['Log Returns'] = np.log(sp500['Returns'] + 1)
sp500['CumLog Returns'] = (sp500['Log Returns']).cumsum()

sp500['Volatility'] = sp500['Returns'].rolling(window=151).std() * np.sqrt(252)

sns.lineplot(y='Volatility', x=sp500.index, data=sp500, ci=None, estimator=None, label='6-months volatility',
             color='red')
plt.axhline(y=np.mean(sp500['Volatility']), color='gray', linestyle='--', label = 'Average Volatility', lw=2, alpha=0.7)
plt.xlim(dt.datetime(1990,1 , 1), dt.datetime(2021,1,1))
plt.title('S&P500 6-months rolling volatility ')
plt.ylabel('Volatility')
plt.xlabel('')
plt.legend()
plt.savefig('SP500_vol.pdf')
plt.show()
plt.close()


current_max = np.maximum.accumulate(sp500['Cumulative Returns'])
underwater = -100 * ((current_max - sp500['Cumulative Returns']) / current_max)
g = underwater.plot(kind='area', color='tomato', alpha=0.7)
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
plt.grid(True)
plt.ylabel('Drawdown')
plt.title('Underwater Plot of the S&P500 index')
plt.xlabel('')
plt.savefig('SP500_underwater.pdf')
plt.show()


annual_returns = pd.DataFrame(ep.aggregate_returns(sp500['Returns'],'yearly'))
ax = plt.gca()
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
ax.axhline(
        100 *
        annual_returns.values.mean(),
        color='gray',
        linestyle='--',
        lw=2,
        alpha=0.7)
(100 * annual_returns.sort_index(ascending=True)
).plot(ax=ax, kind='bar', alpha=1)
ax.axhline(0.0, color='black', linestyle='-', lw=3)
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
ax.set_xlabel('')
ax.set_ylabel('Returns')
ax.set_title("Annual returns of the S&P 500 index")
ax.legend(['Mean'], frameon=True, framealpha=1)
ax.grid(b=True, axis='y')
plt.savefig('SP500_annual_ret.pdf')
plt.show()

cagr = ep.annual_return(sp500['Returns'])
sharpe = ep.sharpe_ratio(sp500['Returns'])